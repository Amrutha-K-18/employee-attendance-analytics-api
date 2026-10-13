"""
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB   (a local .env is loaded for convenience; real env vars win)

Everything lives in this single file, in this order:
  1. configuration & database
  2. pure helpers (time zone, rules R1-R5, rounding, serialisation)
  3. request models
  4. indexes
  5. endpoints: system, employees, attendance, analytics, admin/explain
"""
import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Literal, Optional, Tuple

from bson import Decimal128, json_util
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError, OperationFailure

load_dotenv()  # does NOT override variables that are already set in the environment

# --------------------------------------------------------------------------- #
# 1. Configuration & database
# --------------------------------------------------------------------------- #
UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# tz_aware=True: PyMongo hands back aware UTC datetimes instead of naive ones.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    tzinfo=UTC,
    serverSelectionTimeoutMS=4000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

PRESENCE = ("PRESENT", "WFH", "ON_DUTY")  # R6
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"
HHMM_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"
MS_MIN, MS_MAX = 100_000_000_000, 4_102_444_800_000  # contract bounds for epoch millis

_indexes_ready = False


# --------------------------------------------------------------------------- #
# 2. Helpers
# --------------------------------------------------------------------------- #
def invalid(msg: str, *loc: str):
    """Raise a 422 in FastAPI's standard validation-error shape ({"detail": [ ... ]})."""
    raise RequestValidationError(
        [{"type": "value_error", "loc": loc or ("body",), "msg": msg, "input": None}]
    )


def parse_date(value: str, *loc: str) -> date:
    """Strict YYYY-MM-DD (date.fromisoformat alone also accepts 20260701 etc.)."""
    if not DATE_RE.match(value):
        invalid("date must be YYYY-MM-DD", *loc)
    try:
        return date.fromisoformat(value)
    except ValueError:
        invalid("not a valid calendar date", *loc)


def month_bounds(month: str) -> Tuple[date, date]:
    y, m = int(month[:4]), int(month[5:7])
    first = date(y, m, 1)
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return first, nxt - timedelta(days=1)


def half_up(value, places: int) -> float:
    """Round half-up on the exact decimal value (never banker's, never binary-float noise)."""
    d = value if isinstance(value, Decimal) else Decimal(repr(value))
    return float(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def ms_to_dt(ms: int) -> datetime:
    """Epoch millis -> aware UTC datetime truncated to whole seconds (R1)."""
    return datetime.fromtimestamp(ms // 1000, tz=UTC)


def dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:  # defensive: treat naive values as UTC, which is what BSON means
        dt = dt.replace(tzinfo=UTC)
    return (dt - EPOCH) // timedelta(milliseconds=1)


def hhmm(value: str) -> Tuple[int, int]:
    h, m = value.split(":")
    return int(h), int(m)


def is_overnight(shift_start: str, shift_end: str) -> bool:
    return hhmm(shift_end) <= hhmm(shift_start)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """R1: IST calendar date of the punch-in; an overnight shift's early-morning punch-in
    (earlier than shift_end) belongs to the shift that started the previous day."""
    ist = punch_in.astimezone(IST)
    d = ist.date()
    if is_overnight(shift_start, shift_end):
        eh, em = hhmm(shift_end)
        if ist.time() < ist.time().replace(hour=eh, minute=em, second=0, microsecond=0):
            d -= timedelta(days=1)
    return d.isoformat()


def shift_bounds(date_str: str, shift_start: str, shift_end: str) -> Tuple[datetime, datetime]:
    """Shift start/end instants (aware, IST) for the attendance date."""
    d = date.fromisoformat(date_str)
    sh, sm = hhmm(shift_start)
    eh, em = hhmm(shift_end)
    start = datetime(d.year, d.month, d.day, sh, sm, tzinfo=IST)
    end = datetime(d.year, d.month, d.day, eh, em, tzinfo=IST)
    if is_overnight(shift_start, shift_end):
        end += timedelta(days=1)
    return start, end


def compute_late_minutes(punch_in: datetime, date_str: str, shift_start: str, shift_end: str) -> int:
    """R2: late only if strictly more than 10:00 after shift start; then floor(minutes since start)."""
    start, _ = shift_bounds(date_str, shift_start, shift_end)
    seconds = int((punch_in - start).total_seconds())
    return seconds // 60 if seconds > 600 else 0


def compute_overtime(punch_out: datetime, date_str: str, shift_start: str, shift_end: str) -> int:
    """R3: floor(minutes) from shift end to punch-out, counted only when >= 30."""
    _, end = shift_bounds(date_str, shift_start, shift_end)
    minutes = int((punch_out - end).total_seconds()) // 60
    return minutes if minutes >= 30 else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600 rounded to 2 decimals half-up."""
    seconds = int((punch_out - punch_in).total_seconds())
    return half_up(Decimal(seconds) / Decimal(3600), 2)


def compute_half_day(work_hours: Optional[float]) -> bool:
    """R5: rounded work hours < 4.50."""
    return work_hours is not None and Decimal(repr(work_hours)) < Decimal("4.50")


def derive(status: str, punch_in: Optional[datetime], punch_out: Optional[datetime],
           date_str: str, shift_start: str, shift_end: str) -> Dict[str, Any]:
    """All derived fields for one record (used by punch-out and PATCH)."""
    out = {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    if status in PRESENCE and punch_in is not None:
        out["late_minutes"] = compute_late_minutes(punch_in, date_str, shift_start, shift_end)
        if punch_out is not None:
            out["work_hours"] = compute_work_hours(punch_in, punch_out)
            out["overtime_minutes"] = compute_overtime(punch_out, date_str, shift_start, shift_end)
            out["half_day"] = compute_half_day(out["work_hours"])
    return out


def _hist_value(v):
    return dt_to_ms(v) if isinstance(v, datetime) else v


def record_out(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Stored attendance document -> API record (epoch millis, defaults for legacy gaps, no _id)."""
    history = []
    for h in doc.get("history") or []:
        history.append({
            "at": dt_to_ms(h["at"]),
            "by": h.get("by"),
            "reason": h.get("reason"),
            "changes": {k: {"from": _hist_value(v.get("from")), "to": _hist_value(v.get("to"))}
                        for k, v in (h.get("changes") or {}).items()},
        })
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc["punch_in"]) if doc.get("punch_in") else None,
        "punch_out": dt_to_ms(doc["punch_out"]) if doc.get("punch_out") else None,
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": history,
    }


def employee_out(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "emp_code": doc["emp_code"], "name": doc["name"], "email": doc["email"],
        "department": doc["department"], "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"], "joined_on": doc["joined_on"],
        "created_at": dt_to_ms(doc["created_at"]) if doc.get("created_at") else None,
    }


def get_employee(emp_code: str) -> Optional[Dict[str, Any]]:
    return db.employees.find_one({"emp_code": emp_code})


def not_found(msg: str):
    return JSONResponse(status_code=404, content={"detail": msg})


# --------------------------------------------------------------------------- #
# 3. Request models
# --------------------------------------------------------------------------- #
def epoch_ms():
    """Optional epoch-millis field; StrictInt rejects floats, strings and bools."""
    return Field(default=None, ge=MS_MIN, le=MS_MAX)


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=HHMM_PATTERN)
    shift_end: str = Field(default="18:30", pattern=HHMM_PATTERN)
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def _joined_on(cls, v: str) -> str:
        if not DATE_RE.match(v):
            raise ValueError("joined_on must be YYYY-MM-DD")
        date.fromisoformat(v)  # ValueError -> 422
        return v

    @model_validator(mode="after")
    def _shift(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[StrictInt] = epoch_ms()
    status: PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[StrictInt] = epoch_ms()


class RegularizeIn(BaseModel):
    status: Optional[Status] = None
    punch_in: Optional[StrictInt] = epoch_ms()
    punch_out: Optional[StrictInt] = epoch_ms()
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# 4. Indexes (created at startup, idempotent)
# --------------------------------------------------------------------------- #
def ensure_indexes() -> None:
    global _indexes_ready
    emp, log = db.employees, db.attendance_logs
    # employees
    emp.create_index([("emp_code", 1)], unique=True, name="uq_emp_code")        # identity + 409 on duplicate
    emp.create_index([("department", 1), ("emp_code", 1)], name="dept_emp")     # filtered, sorted list
    emp.create_index([("department", 1), ("joined_on", 1)], name="dept_joined")  # headcount / trend
    emp.create_index([("joined_on", 1)], name="joined")                          # headcount without dept filter
    # attendance_logs
    log.create_index([("emp_code", 1), ("date", 1)], unique=True, name="uq_emp_date")  # one record/day; monthly; lookups
    log.create_index([("date", -1), ("emp_code", 1)], name="date_emp")                 # list sort; month scans
    log.create_index([("status", 1), ("date", -1), ("emp_code", 1)], name="status_date_emp")  # status filter + sort
    log.create_index([("emp_code", 1), ("punch_in", -1)], name="emp_punch_in")         # punch-out lookup
    _indexes_ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Mongo may still be starting; retry briefly so /health can come up inside the 20s window.
    deadline = time.time() + 8
    while True:
        try:
            ensure_indexes()
            break
        except PyMongoError:
            if time.time() > deadline:
                break  # /health stays 503 and retries ensure_indexes() once Mongo answers
            time.sleep(1)
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


@app.exception_handler(PyMongoError)
async def mongo_error_handler(_: Request, exc: PyMongoError):
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


# --------------------------------------------------------------------------- #
# 5a. System
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        if not _indexes_ready:
            ensure_indexes()
    except PyMongoError:
        return JSONResponse(status_code=503, content={"detail": "database unavailable"})
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# 5b. Employees
# --------------------------------------------------------------------------- #
@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_utc()
    try:
        db.employees.insert_one(doc)  # the unique index is the only arbiter of duplicates
    except DuplicateKeyError:
        return JSONResponse(status_code=409, content={"detail": "emp_code already exists"})
    return employee_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q: Dict[str, Any] = {"department": department} if department else {}
    total = db.employees.count_documents(q)
    cur = db.employees.find(q).sort("emp_code", 1).skip((page - 1) * page_size).limit(page_size)
    return {"items": [employee_out(d) for d in cur], "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# 5c. Attendance
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee(body.emp_code)
    if not emp:
        return not_found("employee not found")
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    d = attendance_date(ts, emp["shift_start"], emp["shift_end"])
    doc = {
        "emp_code": body.emp_code,
        "date": d,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(ts, d, emp["shift_start"], emp["shift_end"]),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date): exactly one racer wins
    except DuplicateKeyError:
        return JSONResponse(status_code=409, content={"detail": "already punched in for this date"})
    return record_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee(body.emp_code)
    if not emp:
        return not_found("employee not found")
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    # most recent record whose punch_in is at or before the punch-out (works across midnight)
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}}, sort=[("punch_in", -1)]
    )
    if not rec or rec.get("punch_out") is not None:
        latest_open = db.attendance_logs.find_one(
            {"emp_code": body.emp_code, "punch_in": {"$ne": None}, "punch_out": None},
            sort=[("punch_in", -1)],
        )
        if latest_open is not None and ts <= latest_open["punch_in"]:
            invalid("punched_at must be after punch_in", "body", "punched_at")
    if not rec:
        return not_found("no punch-in found")
    if rec.get("punch_out") is not None:
        return JSONResponse(status_code=409, content={"detail": "already punched out"})
    p_in = rec["punch_in"]
    if ts <= p_in:
        invalid("punched_at must be after punch_in", "body", "punched_at")
    if ts - p_in > timedelta(hours=24):
        invalid("punched_at is more than 24 hours after punch_in", "body", "punched_at")
    d = derive(rec["status"], p_in, ts, rec["date"], emp["shift_start"], emp["shift_end"])
    # compare-and-set on "still open": of two simultaneous punch-outs only one matches
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {"$set": {"punch_out": ts, **d}},
        return_document=True,
    )
    if updated is None:
        return JSONResponse(status_code=409, content={"detail": "already punched out"})
    return record_out(updated)


def build_attendance_query(emp_code, date_from, date_to, status, page, page_size):
    """Shared by GET /attendance and its explain: (filter, sort, skip, limit)."""
    f = parse_date(date_from, "query", "date_from") if date_from else None
    t = parse_date(date_to, "query", "date_to") if date_to else None
    if f and t and f > t:
        invalid("date_from must not be after date_to", "query", "date_from")
    q: Dict[str, Any] = {}
    if emp_code:
        q["emp_code"] = emp_code
    if f or t:
        q["date"] = {}
        if f:
            q["date"]["$gte"] = date_from
        if t:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status
    return q, {"date": -1, "emp_code": 1}, (page - 1) * page_size, page_size


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q, sort, skip, limit = build_attendance_query(emp_code, date_from, date_to, status, page, page_size)
    total = db.attendance_logs.count_documents(q) if q else db.attendance_logs.estimated_document_count()
    cur = db.attendance_logs.find(q).sort(list(sort.items())).skip(skip).limit(limit)
    return {"items": [record_out(d) for d in cur], "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: str, body: RegularizeIn):
    parse_date(date, "path", "date")
    emp = get_employee(emp_code)
    if not emp:
        return not_found("employee not found")
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not rec:
        return not_found("no record for that date")

    old = {
        "status": rec["status"],
        "punch_in": rec.get("punch_in"),
        "punch_out": rec.get("punch_out"),
        "work_hours": rec.get("work_hours"),
        "late_minutes": rec.get("late_minutes") or 0,
        "overtime_minutes": rec.get("overtime_minutes") or 0,
        "half_day": bool(rec.get("half_day", False)),
    }
    new_status = body.status or old["status"]
    req_in = ms_to_dt(body.punch_in) if body.punch_in is not None else None
    req_out = ms_to_dt(body.punch_out) if body.punch_out is not None else None

    if new_status in ("ABSENT", "LEAVE"):
        if req_in is not None or req_out is not None:
            invalid(f"{new_status} clears punch times; do not supply punch_in/punch_out", "body", "status")
        new_in = new_out = None
    else:
        new_in = req_in or old["punch_in"]
        new_out = req_out or old["punch_out"]
        if new_in is None:
            invalid("a presence status requires a punch_in", "body", "punch_in")
        if attendance_date(new_in, emp["shift_start"], emp["shift_end"]) != date:
            invalid("punch_in must stay on the record's attendance date", "body", "punch_in")
        if new_out is not None:
            if new_out <= new_in:
                invalid("punch_out must be after punch_in", "body", "punch_out")
            if new_out - new_in > timedelta(hours=24):
                invalid("punch_out must be within 24 hours of punch_in", "body", "punch_out")

    new = {"status": new_status, "punch_in": new_in, "punch_out": new_out,
           **derive(new_status, new_in, new_out, date, emp["shift_start"], emp["shift_end"])}
    changes = {k: {"from": old[k], "to": new[k]} for k in old if old[k] != new[k]}
    if not changes:
        invalid("the request changes nothing", "body")

    now = datetime.now(UTC)
    entry = {"at": now.replace(microsecond=now.microsecond // 1000 * 1000),  # BSON keeps milliseconds
             "by": body.regularized_by, "reason": body.reason, "changes": changes}
    # Optimistic concurrency: only apply if the record is exactly as we read it (including the
    # history length). A concurrent edit makes this miss -> 409, so no history entry is ever lost.
    guard = {
        "emp_code": emp_code, "date": date,
        "status": rec["status"], "punch_in": rec.get("punch_in"), "punch_out": rec.get("punch_out"),
        "$expr": {"$eq": [{"$size": {"$ifNull": ["$history", []]}}, len(rec.get("history") or [])]},
    }
    updated = db.attendance_logs.find_one_and_update(
        guard, {"$set": new, "$push": {"history": entry}}, return_document=True
    )
    if updated is None:
        return JSONResponse(status_code=409, content={"detail": "record was modified concurrently; retry"})
    return record_out(updated)


# --------------------------------------------------------------------------- #
# 5d. Analytics - pipeline builders (shared by the endpoints and /admin/explain)
# --------------------------------------------------------------------------- #
def _f(prefix: str, name: str) -> str:
    return f"{prefix}{name}"


def _is_present(prefix: str) -> Dict[str, Any]:
    return {"$in": [_f(prefix, "status"), list(PRESENCE)]}


def _late_minutes(prefix: str) -> Dict[str, Any]:
    return {"$ifNull": [_f(prefix, "late_minutes"), 0]}


def _present_record_value(prefix: str) -> Dict[str, Any]:
    """Presence value of a record (R5-R6): 0.5 for half_day else 1 when present-status, 0 otherwise."""
    half = {"$cond": [{"$eq": [{"$ifNull": [_f(prefix, "half_day"), False]}, True]}, 0.5, 1]}
    return {"$cond": [_is_present(prefix), half, 0]}


def _present_value(prefix: str) -> Dict[str, Any]:
    """Present-day value of a record (R5-R7): 0 unless present AND Mon-Fri; half day = 0.5."""
    weekday = {"$in": [{"$dayOfWeek": {"$dateFromString": {"dateString": _f(prefix, "date")}}}, [2, 3, 4, 5, 6]]}
    return {"$cond": [weekday, _present_record_value(prefix), 0]}


def _stats(prefix: str) -> Dict[str, Any]:
    """Accumulators shared by the monthly and department summaries. Reads STORED derived values."""
    has_hours = {"$in": [{"$type": _f(prefix, "work_hours")}, ["double", "int", "long", "decimal"]]}
    counted = {"$and": [_is_present(prefix), has_hours]}
    return {
        "present_days": {"$sum": _present_value(prefix)},
        "late_count": {"$sum": {"$cond": [{"$gt": [_late_minutes(prefix), 0]}, 1, 0]}},
        "total_late_minutes": {"$sum": _late_minutes(prefix)},
        "total_overtime_minutes": {"$sum": {"$ifNull": [_f(prefix, "overtime_minutes"), 0]}},
        "leave_count": {"$sum": {"$cond": [{"$eq": [_f(prefix, "status"), "LEAVE"]}, 1, 0]}},
        "on_duty_count": {"$sum": {"$cond": [{"$eq": [_f(prefix, "status"), "ON_DUTY"]}, 1, 0]}},
        # exact decimal sum of work_hours (avoids binary-float drift); divided in Python with Decimal
        "wh_sum": {"$sum": {"$cond": [counted, {"$toDecimal": _f(prefix, "work_hours")}, 0]}},
        "wh_count": {"$sum": {"$cond": [counted, 1, 0]}},
    }


def build_monthly_pipeline(emp_code: str, month: str) -> List[Dict[str, Any]]:
    first, last = month_bounds(month)
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
        {"$group": {"_id": None, **_stats("$")}},
    ]


def build_department_summary_pipeline(month: str, department: Optional[str]) -> List[Dict[str, Any]]:
    """Starts from employees (so people with no logs are counted, later joiners are not), joins the
    month's logs through the (emp_code, date) index, and groups by department."""
    first, last = month_bounds(month)
    match: Dict[str, Any] = {"joined_on": {"$lte": last.isoformat()}}  # R9
    if department:
        match["department"] = department
    return [
        {"$match": match},
        {"$lookup": {
            "from": "attendance_logs", "localField": "emp_code", "foreignField": "emp_code",
            "pipeline": [{"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}}],
            "as": "logs",
        }},
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},  # keep employees with zero logs
        {"$group": {"_id": "$department", "emps": {"$addToSet": "$emp_code"}, **_stats("$logs.")}},
        {"$addFields": {"headcount": {"$size": "$emps"}}},
        {"$sort": {"_id": 1}},
    ]


def build_leaderboard_pipeline(month: str, limit: int, department: Optional[str]) -> List[Dict[str, Any]]:
    first, last = month_bounds(month)
    pipeline: List[Dict[str, Any]] = [
        {"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
        {"$group": {
            "_id": "$emp_code",
            "total_late_minutes": {"$sum": _late_minutes("$")},
            "late_count": {"$sum": {"$cond": [{"$gt": [_late_minutes("$"), 0]}, 1, 0]}},
        }},
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "e"}},
        {"$unwind": "$e"},  # logs whose employee does not exist are dropped
    ]
    if department:
        pipeline.append({"$match": {"e.department": department}})  # rank inside the department only
    pipeline += [
        # standard competition ranking: ties share a rank, the next rank is skipped (1, 2, 2, 4)
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},  # limit applies AFTER ranking
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$project": {"_id": 0, "rank": 1, "emp_code": "$_id", "name": "$e.name",
                      "department": "$e.department", "total_late_minutes": 1, "late_count": 1}},
    ]
    return pipeline


def load_department(department: str) -> Tuple[List[str], List[str]]:
    """(emp_codes, joined_on strings) of a department; 404 handled by the caller if empty."""
    docs = list(db.employees.find({"department": department}, {"_id": 0, "emp_code": 1, "joined_on": 1}))
    return [d["emp_code"] for d in docs], [d["joined_on"] for d in docs]


def build_trend_pipeline(codes: List[str], joined: List[str], d_from: date, d_to: date) -> List[Dict[str, Any]]:
    n_days = (d_to - d_from).days + 1
    f, t = d_from.isoformat(), d_to.isoformat()
    return [
        {"$match": {"emp_code": {"$in": codes}, "date": {"$gte": f, "$lte": t}}},
        # $facet always emits exactly one document, even when no record matched, so the calendar
        # below is generated no matter what the data looks like.
        {"$facet": {"days": [{"$group": {
            "_id": "$date",
            "present_count": {"$sum": _present_record_value("$")},
            "late_count": {"$sum": {"$cond": [{"$gt": [_late_minutes("$"), 0]}, 1, 0]}},
        }}]}},
        # gap filling: one row per calendar day from..to
        {"$project": {"_id": 0, "rows": {"$map": {
            "input": {"$range": [0, n_days]}, "as": "i",
            "in": {"$let": {
                "vars": {"day": {"$dateAdd": {"startDate": {"$dateFromString": {"dateString": f}},
                                              "unit": "day", "amount": "$$i"}}},
                "in": {"$let": {
                    "vars": {"ds": {"$dateToString": {"date": "$$day", "format": "%Y-%m-%d"}}},
                    "in": {
                        "date": "$$ds",
                        "dow": {"$dayOfWeek": "$$day"},
                        "st": {"$arrayElemAt": [{"$filter": {
                            "input": "$days", "as": "x", "cond": {"$eq": ["$$x._id", "$$ds"]}}}, 0]},
                    },
                }},
            }},
        }}}},
        {"$unwind": "$rows"},
        {"$replaceRoot": {"newRoot": "$rows"}},
        {"$addFields": {
            "is_working_day": {"$in": ["$dow", [2, 3, 4, 5, 6]]},
            "headcount": {"$size": {"$filter": {  # R9: employees who have joined by that day
                "input": {"$literal": joined}, "as": "j", "cond": {"$lte": ["$$j", "$date"]}}}},
            "present_count": {"$ifNull": ["$st.present_count", 0]},
            "late_count": {"$ifNull": ["$st.late_count", 0]},
        }},
        # rate in 1/10000 units, half-up, using exact integer-valued arithmetic:
        # floor(x*10000 + 1/2) with x = p/h  ==  floor((20000*p + h) / (2*h))
        {"$addFields": {"rate_int": {"$cond": [
            {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
            {"$floor": {"$divide": [{"$add": [{"$multiply": [20000, "$present_count"]}, "$headcount"]},
                                    {"$multiply": [2, "$headcount"]}]}},
            None]}}},
        # moving window over the current row and up to 6 before it, inside the requested range only
        {"$setWindowFields": {"sortBy": {"date": 1}, "output": {
            "win_sum": {"$sum": "$rate_int", "window": {"documents": [-6, 0]}},
            "win_cnt": {"$sum": {"$cond": [{"$ne": ["$rate_int", None]}, 1, 0]}, "window": {"documents": [-6, 0]}},
        }}},
        {"$addFields": {
            "attendance_rate": {"$cond": [{"$eq": ["$rate_int", None]}, None, {"$divide": ["$rate_int", 10000]}]},
            # mean of the (rounded) rates, again half-up to 4 decimals: floor((2S + n) / 2n) / 10000
            "moving_avg_7d": {"$cond": [{"$gt": ["$win_cnt", 0]}, {"$divide": [{"$floor": {"$divide": [
                {"$add": [{"$multiply": [2, "$win_sum"]}, "$win_cnt"]},
                {"$multiply": [2, "$win_cnt"]}]}}, 10000]}, None]},
        }},
        {"$sort": {"date": 1}},
        {"$project": {"_id": 0, "date": 1, "is_working_day": 1, "headcount": 1, "present_count": 1,
                      "late_count": 1, "attendance_rate": 1, "moving_avg_7d": 1}},
    ]


# --------------------------------------------------------------------------- #
# 5e. Analytics - endpoints
# --------------------------------------------------------------------------- #
def working_days(first: date, last: date, joined_on: str) -> int:
    """R7: Mon-Fri in [max(first, joined_on), last]."""
    start = max(first, date.fromisoformat(joined_on))
    n, d = 0, start
    while d <= last:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


def _num(v) -> Decimal:
    return v.to_decimal() if isinstance(v, Decimal128) else Decimal(repr(v)) if v is not None else Decimal(0)


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(pattern=MONTH_PATTERN)):
    emp = get_employee(emp_code)
    if not emp:
        return not_found("employee not found")
    first, last = month_bounds(month)
    rows = list(db.attendance_logs.aggregate(build_monthly_pipeline(emp_code, month)))
    r = rows[0] if rows else {}
    wd = working_days(first, last, emp["joined_on"])
    present = float(r.get("present_days", 0) or 0)
    pct = None
    if wd > 0:
        pct = half_up(Decimal(repr(present)) / Decimal(wd) * 100, 2)
    return {
        "emp_code": emp_code, "month": month, "working_days": wd,
        "present_days": half_up(present, 2),
        "leave_days": int(r.get("leave_count", 0)),
        "late_count": int(r.get("late_count", 0)),
        "total_late_minutes": int(r.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(r.get("total_overtime_minutes", 0)),
        "attendance_pct": pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(pattern=MONTH_PATTERN), department: Optional[str] = None):
    items = []
    for r in db.employees.aggregate(build_department_summary_pipeline(month, department)):
        avg = None
        if r["wh_count"] > 0:
            avg = half_up(_num(r["wh_sum"]) / Decimal(r["wh_count"]), 2)
        items.append({
            "department": r["_id"], "headcount": int(r["headcount"]),
            "present_days": half_up(float(r["present_days"]), 2), "avg_work_hours": avg,
            "late_count": int(r["late_count"]), "total_late_minutes": int(r["total_late_minutes"]),
            "leave_count": int(r["leave_count"]), "on_duty_count": int(r["on_duty_count"]),
        })
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: str = Query(pattern=MONTH_PATTERN), limit: int = Query(10, ge=1, le=50),
                     department: Optional[str] = None):
    rows = list(db.attendance_logs.aggregate(build_leaderboard_pipeline(month, limit, department)))
    items = [{"rank": int(r["rank"]), "emp_code": r["emp_code"], "name": r["name"],
              "department": r["department"], "total_late_minutes": int(r["total_late_minutes"]),
              "late_count": int(r["late_count"])} for r in rows]
    return {"month": month, "items": items}


def prepare_trend(department: str, d_from_s: str, d_to_s: str):
    d_from = parse_date(d_from_s, "query", "from")
    d_to = parse_date(d_to_s, "query", "to")
    if d_to < d_from:
        invalid("to must not be before from", "query", "to")
    if (d_to - d_from).days + 1 > 92:
        invalid("range must not exceed 92 days", "query", "to")
    codes, joined = load_department(department)
    return codes, joined, d_from, d_to


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str, from_: str = Query(alias="from"), to: str = Query()):
    codes, joined, d_from, d_to = prepare_trend(department, from_, to)
    if not codes:
        return not_found("unknown department")
    rows = list(db.attendance_logs.aggregate(build_trend_pipeline(codes, joined, d_from, d_to)))
    items = [{
        "date": r["date"], "is_working_day": bool(r["is_working_day"]), "headcount": int(r["headcount"]),
        "present_count": float(r["present_count"]), "late_count": int(r["late_count"]),
        "attendance_rate": r["attendance_rate"], "moving_avg_7d": r["moving_avg_7d"],
    } for r in rows]
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# 5f. Admin: explain
# --------------------------------------------------------------------------- #
ExplainTarget = Literal["attendance_list", "employee_monthly", "department_summary",
                        "late_leaderboard", "department_trend"]


def _require(**params):
    for name, value in params.items():
        if value is None:
            invalid(f"query parameter '{name.rstrip('_')}' is required for this endpoint", "query", name.rstrip("_"))


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: ExplainTarget,
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=MONTH_PATTERN),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        q, sort, skip, lim = build_attendance_query(emp_code, date_from, date_to, status, page, page_size)
        coll = "attendance_logs"
        cmd = {"find": coll, "filter": q, "sort": sort, "skip": skip, "limit": lim}
    else:
        if endpoint == "employee_monthly":
            _require(emp_code=emp_code, month=month)
            coll, pipeline = "attendance_logs", build_monthly_pipeline(emp_code, month)
        elif endpoint == "department_summary":
            _require(month=month)
            coll, pipeline = "employees", build_department_summary_pipeline(month, department)
        elif endpoint == "late_leaderboard":
            _require(month=month)
            coll, pipeline = "attendance_logs", build_leaderboard_pipeline(month, limit, department)
        else:  # department_trend
            _require(department=department, from_=from_, to=to)
            codes, joined, d_from, d_to = prepare_trend(department, from_, to)
            if not codes:
                return not_found("unknown department")
            coll, pipeline = "attendance_logs", build_trend_pipeline(codes, joined, d_from, d_to)
        cmd = {"aggregate": coll, "pipeline": pipeline, "cursor": {}}
    try:
        out = db.command({"explain": cmd, "verbosity": "executionStats"})
    except OperationFailure as exc:
        return JSONResponse(status_code=500, content={"detail": f"explain failed: {exc}"})
    return {"endpoint": endpoint, "collection": coll, "explain": json.loads(json_util.dumps(out))}
