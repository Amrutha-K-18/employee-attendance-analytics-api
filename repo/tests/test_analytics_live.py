"""Analytics + explain tests. These need a REAL MongoDB (6.0+) because the pipelines use
$setWindowFields, $facet, $dateAdd ... which mongomock cannot run.

    docker run -d -p 27017:27017 mongo:7
    MONGO_URI=mongodb://localhost:27017 MONGO_DB=attendance_test pytest tests/test_analytics_live.py

The database name must end in "_test": the tests DROP it.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("MONGO_DB", "attendance_test")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pymongo import MongoClient  # noqa: E402

import app.main as m  # noqa: E402

TEST_DB_NAME = os.getenv("MONGO_DB", "attendance_test")
pytestmark = pytest.mark.skipif(not TEST_DB_NAME.endswith("_test"), reason="MONGO_DB must end with _test")


def E(code, dept, joined, **kw):
    return {"emp_code": code, "name": f"Name {code}", "email": f"{code}@x.com", "department": dept,
            "shift_start": "09:30", "shift_end": "18:30", "joined_on": joined,
            "created_at": m.now_utc(), **kw}


def L(code, d, status="PRESENT", late=0, ot=0, wh=None, half=False):
    return {"emp_code": code, "date": d, "status": status, "punch_in": None, "punch_out": None,
            "work_hours": wh, "late_minutes": late, "overtime_minutes": ot, "half_day": half, "history": []}


@pytest.fixture(scope="module")
def client():
    prev_client, prev_db, prev_ready = m.client, m.db, m._indexes_ready
    m.client = MongoClient(
        os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        tz_aware=True,
        tzinfo=m.UTC,
        serverSelectionTimeoutMS=4000,
    )
    m.db = m.client[TEST_DB_NAME]
    m._indexes_ready = False
    try:
        m.client.admin.command("ping")
    except Exception:  # pragma: no cover
        m.client, m.db, m._indexes_ready = prev_client, prev_db, prev_ready
        pytest.skip("MongoDB not reachable")
    m.client.drop_database(m.db.name)
    try:
        with TestClient(m.app) as c:  # runs ensure_indexes() via lifespan
            m.db.employees.insert_many([
                E("EMP0001", "Eng", "2026-01-05"), E("EMP0002", "Eng", "2026-07-15"),   # EMP0002: mid-month joiner
                E("EMP0003", "Eng", "2026-01-01"),                                      # no logs at all
                E("EMP0004", "Eng", "2026-08-01"),                                      # joins after July
                E("EMP0005", "Ops", "2026-01-01"), E("EMP0006", "Ops", "2026-01-01"),
            ])
            m.db.attendance_logs.insert_many([
                L("EMP0001", "2026-07-06", late=35, ot=40, wh=9.08),
                L("EMP0001", "2026-07-07", wh=3.5, half=True),
                L("EMP0001", "2026-07-08", status="LEAVE"),
                L("EMP0001", "2026-07-09", status="ABSENT"),
                L("EMP0001", "2026-07-11", late=20, ot=30, wh=8.0),         # Saturday
                L("EMP0002", "2026-07-15", late=12, wh=9.0),
                L("EMP0002", "2026-07-16", status="ON_DUTY", wh=8.5),
                L("EMP0005", "2026-07-06", late=35, wh=9.0),
                L("EMP0006", "2026-07-07", late=35, wh=9.0),
                L("EMP9999", "2026-07-06", late=500),                        # employee does not exist
            ])
            yield c
    finally:
        m.client.drop_database(m.db.name)
        m.client, m.db, m._indexes_ready = prev_client, prev_db, prev_ready


def test_indexes_exist(client):
    names = {i["name"] for i in m.db.attendance_logs.list_indexes()}
    assert {"uq_emp_date", "date_emp", "emp_punch_in"} <= names


def test_employee_monthly(client):
    j = client.get("/analytics/employees/EMP0001/monthly?month=2026-07").json()
    assert j == {"emp_code": "EMP0001", "month": "2026-07", "working_days": 23, "present_days": 1.5,
                 "leave_days": 1, "late_count": 2, "total_late_minutes": 55, "total_overtime_minutes": 70,
                 "attendance_pct": 6.52}
    j = client.get("/analytics/employees/EMP0002/monthly?month=2026-07").json()
    assert j["working_days"] == 13 and j["present_days"] == 2 and j["attendance_pct"] == 15.38
    j = client.get("/analytics/employees/EMP0003/monthly?month=2026-07").json()
    assert j["present_days"] == 0 and j["attendance_pct"] == 0
    assert client.get("/analytics/employees/EMP0004/monthly?month=2026-07").json()["attendance_pct"] is None
    assert client.get("/analytics/employees/NOPE/monthly?month=2026-07").status_code == 404
    assert client.get("/analytics/employees/EMP0001/monthly?month=2026-13").status_code == 422


def test_department_summary(client):
    j = client.get("/analytics/departments/summary?month=2026-07").json()
    eng, ops = j["items"]
    assert eng == {"department": "Eng", "headcount": 3, "present_days": 3.5, "avg_work_hours": 7.62,
                   "late_count": 3, "total_late_minutes": 67, "leave_count": 1, "on_duty_count": 1}
    assert ops == {"department": "Ops", "headcount": 2, "present_days": 2.0, "avg_work_hours": 9.0,
                   "late_count": 2, "total_late_minutes": 70, "leave_count": 0, "on_duty_count": 0}
    assert len(client.get("/analytics/departments/summary?month=2026-07&department=Ops").json()["items"]) == 1
    assert client.get("/analytics/departments/summary?month=2025-01").json()["items"] == []  # headcount 0 -> omitted
    aug = client.get("/analytics/departments/summary?month=2026-08").json()["items"]
    assert aug[0]["headcount"] == 4 and aug[0]["avg_work_hours"] is None  # EMP0004 has joined by 31 Aug


def test_leaderboard_ties(client):
    rows = client.get("/analytics/leaderboard/late?month=2026-07&limit=2").json()["items"]
    assert [(r["rank"], r["emp_code"], r["total_late_minutes"]) for r in rows] == [
        (1, "EMP0001", 55), (2, "EMP0005", 35), (2, "EMP0006", 35)]        # tie at the cutoff: both returned
    rows = client.get("/analytics/leaderboard/late?month=2026-07&limit=10").json()["items"]
    assert [r["rank"] for r in rows] == [1, 2, 2, 4] and rows[-1]["emp_code"] == "EMP0002"  # 1,2,2,4 and no orphan
    rows = client.get("/analytics/leaderboard/late?month=2026-07&department=Ops").json()["items"]
    assert [(r["rank"], r["emp_code"]) for r in rows] == [(1, "EMP0005"), (1, "EMP0006")]
    assert client.get("/analytics/leaderboard/late?month=2026-07&limit=51").status_code == 422


def test_trend_gap_filling_and_moving_average(client):
    j = client.get("/analytics/departments/Eng/trend?from=2026-07-06&to=2026-07-12").json()["items"]
    assert [r["date"] for r in j] == [f"2026-07-{d:02d}" for d in range(6, 13)]
    assert [r["is_working_day"] for r in j] == [True] * 5 + [False] * 2
    assert [r["headcount"] for r in j] == [2] * 7
    assert [r["present_count"] for r in j] == [1, 0.5, 0, 0, 0, 1, 0]
    assert [r["late_count"] for r in j] == [1, 0, 0, 0, 0, 1, 0]
    assert [r["attendance_rate"] for r in j] == [0.5, 0.25, 0.0, 0.0, 0.0, None, None]
    assert [r["moving_avg_7d"] for r in j] == [0.5, 0.375, 0.25, 0.1875, 0.15, 0.15, 0.15]
    j = client.get("/analytics/departments/Eng/trend?from=2026-07-13&to=2026-07-17").json()["items"]
    assert [r["headcount"] for r in j] == [2, 2, 3, 3, 3]                     # mid-month joiner appears on the 15th
    assert [r["attendance_rate"] for r in j] == [0.0, 0.0, 0.3333, 0.3333, 0.0]
    assert [r["moving_avg_7d"] for r in j] == [0.0, 0.0, 0.1111, 0.1667, 0.1333]


def test_trend_with_no_records_and_errors(client):
    j = client.get("/analytics/departments/Ops/trend?from=2026-09-01&to=2026-09-03").json()["items"]
    assert len(j) == 3 and all(r["attendance_rate"] == 0.0 and r["moving_avg_7d"] == 0.0 for r in j)
    assert client.get("/analytics/departments/Nope/trend?from=2026-09-01&to=2026-09-03").status_code == 404
    assert client.get("/analytics/departments/Eng/trend?from=2026-09-03&to=2026-09-01").status_code == 422
    assert client.get("/analytics/departments/Eng/trend?from=2026-01-01&to=2026-04-03").status_code == 422  # 93 days
    assert client.get("/analytics/departments/Eng/trend?from=2026-01-01&to=2026-04-02").status_code == 200  # 92 days
    assert client.get("/analytics/departments/Eng/trend?from=2026-01-01").status_code == 422


@pytest.mark.parametrize("endpoint,qs", [
    ("attendance_list", ""),
    ("attendance_list", "?emp_code=EMP0001&date_from=2026-07-01&date_to=2026-07-31&status=PRESENT&page=2&page_size=5"),
    ("employee_monthly", "?emp_code=EMP0001&month=2026-07"),
    ("department_summary", "?month=2026-07"),
    ("department_summary", "?month=2026-07&department=Eng"),
    ("late_leaderboard", "?month=2026-07&limit=5"),
    ("late_leaderboard", "?month=2026-07&department=Ops"),
    ("department_trend", "?department=Eng&from=2026-07-01&to=2026-07-31"),
])
def test_explain_uses_indexes(client, endpoint, qs):
    r = client.get(f"/admin/explain/{endpoint}{qs}")
    assert r.status_code == 200, r.text
    text = str(r.json()["explain"])
    assert "IXSCAN" in text and "COLLSCAN" not in text, text[:2000]


def test_explain_validation(client):
    assert client.get("/admin/explain/employee_monthly?month=2026-07").status_code == 422
    assert client.get("/admin/explain/department_trend?department=Eng").status_code == 422


def test_concurrent_punch_in_exactly_one_201(client):
    from concurrent.futures import ThreadPoolExecutor
    m.db.employees.insert_one(E("EMP0777", "Eng", "2026-01-01"))
    body = {"emp_code": "EMP0777", "punched_at": 1783312500000}
    with ThreadPoolExecutor(20) as ex:
        codes = list(ex.map(lambda _: client.post("/attendance/punch-in", json=body).status_code, range(20)))
    assert codes.count(201) == 1 and codes.count(409) == 19
