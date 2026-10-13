"""Business-rule and endpoint tests that run WITHOUT a MongoDB server (mongomock stands in).
Run:  pip install pytest mongomock httpx && pytest tests/test_rules.py
The aggregation pipelines need a real server: see tests/test_analytics_live.py."""
from datetime import datetime, timezone, timedelta
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mongomock  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main as m  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))


def ms(y, mo, d, h, mi, s=0, micro=0):
    dt = datetime(y, mo, d, h, mi, s, micro, tzinfo=IST)
    return (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)) // timedelta(milliseconds=1)


@pytest.fixture()
def client():
    prev_client, prev_db, prev_ready = m.client, m.db, m._indexes_ready
    m.client = mongomock.MongoClient(tz_aware=True, tzinfo=m.UTC)
    m.db = m.client["attendance_mock_test"]
    m._indexes_ready = False
    try:
        with TestClient(m.app) as c:
            yield c
    finally:
        m.client, m.db, m._indexes_ready = prev_client, prev_db, prev_ready


def emp(c, code="EMP0001", **kw):
    body = {"emp_code": code, "name": "Asha", "email": "a@x.com", "department": "Eng", "joined_on": "2026-01-05", **kw}
    assert c.post("/employees", json=body).status_code == 201


# ---- pure helpers --------------------------------------------------------------------------
def test_late_rule_boundaries():
    f = lambda h, mi, s: m.compute_late_minutes(datetime(2026, 7, 6, h, mi, s, tzinfo=IST), "2026-07-06", "09:30", "18:30")
    assert f(9, 40, 0) == 0          # exactly 10:00 after start: not late
    assert f(9, 40, 1) == 10         # strictly more than 10:00 -> minutes since shift start
    assert f(10, 15, 59) == 45
    assert f(9, 0, 0) == 0           # early


def test_overtime_threshold():
    f = lambda h, mi: m.compute_overtime(datetime(2026, 7, 6, h, mi, tzinfo=IST), "2026-07-06", "09:30", "18:30")
    assert f(18, 59) == 0 and f(19, 0) == 30 and f(19, 10) == 40 and f(17, 0) == 0


def test_work_hours_half_up_and_half_day():
    a = datetime(2026, 7, 6, 9, 0, tzinfo=IST)
    assert m.compute_work_hours(a, a + timedelta(seconds=32820)) == 9.12     # 9.116666..
    assert m.compute_work_hours(a, a + timedelta(seconds=18)) == 0.01        # 0.005 -> half-up (round() gives 0.01 or 0.0)
    assert m.compute_work_hours(a, a + timedelta(seconds=16200)) == 4.5      # not a half day
    assert m.compute_half_day(4.49) and not m.compute_half_day(4.5)
    assert m.compute_work_hours(a, a + timedelta(seconds=16199)) == 4.5      # rounds to 4.50 first -> not half day


def test_overnight_attendance_date():
    d = lambda y, mo, dd, h, mi: m.attendance_date(datetime(y, mo, dd, h, mi, tzinfo=IST), "22:00", "06:00")
    assert d(2026, 7, 6, 21, 55) == "2026-07-06"
    assert d(2026, 7, 7, 5, 59) == "2026-07-06"     # earlier than shift_end -> previous day's shift
    assert d(2026, 7, 7, 6, 0) == "2026-07-07"
    assert m.attendance_date(datetime(2026, 7, 7, 1, 0, tzinfo=IST), "09:30", "18:30") == "2026-07-07"


# ---- validation ----------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [1_700_000_000, 1.5e12, "1783312500000", -5, 10**14, True])
def test_bad_timestamp_is_422(client, bad):
    emp(client)
    assert client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": bad}).status_code == 422


def test_validation_and_404(client):
    assert client.post("/attendance/punch-in", json={"emp_code": "NOPE", "punched_at": ms(2026, 7, 6, 9, 30)}).status_code == 404
    emp(client)
    for st in ("ABSENT", "LEAVE", "bogus"):
        assert client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "status": st}).status_code == 422
    assert client.get("/attendance?date_from=2026-07-09&date_to=2026-07-01").status_code == 422
    assert client.get("/attendance?date_from=20260701").status_code == 422
    assert client.get("/attendance?page_size=101").status_code == 422
    assert client.get("/attendance?status=NOPE").status_code == 422
    assert client.post("/employees", json={"emp_code": "x", "name": "A", "email": "a@x.com", "department": "E", "joined_on": "2026-02-30"}).status_code == 422


# ---- punch-in / punch-out ------------------------------------------------------------------
def test_punch_in_out_flow(client):
    emp(client)
    r = client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 10, 5, 7, 900_000)})
    assert r.status_code == 201
    j = r.json()
    assert j["date"] == "2026-07-06" and j["late_minutes"] == 35 and j["punch_out"] is None
    assert j["punch_in"] == ms(2026, 7, 6, 10, 5, 7)          # truncated to whole seconds
    assert "id" not in j and "_id" not in j and j["history"] == []
    assert client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 11, 0)}).status_code == 409
    r = client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 19, 10)})
    assert r.status_code == 200
    j = r.json()
    assert j["overtime_minutes"] == 40 and j["work_hours"] == 9.08 and j["half_day"] is False
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 19, 20)}).status_code == 409


def test_punch_out_errors(client):
    emp(client)
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 18, 0)}).status_code == 404
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 9, 30)})
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 9, 0)}).status_code == 422
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 9, 30)}).status_code == 422
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 7, 9, 31)}).status_code == 422  # >24h
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 12, 30)}).json()["half_day"] is True


def test_overnight_shift_across_midnight(client):
    emp(client, "EMP0005", shift_start="22:00", shift_end="06:00")
    j = client.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ms(2026, 7, 6, 21, 55)}).json()
    assert j["date"] == "2026-07-06" and j["late_minutes"] == 0
    j = client.post("/attendance/punch-out", json={"emp_code": "EMP0005", "punched_at": ms(2026, 7, 7, 6, 40)}).json()
    assert j["date"] == "2026-07-06" and j["overtime_minutes"] == 40 and j["work_hours"] == 8.75


# ---- listing -------------------------------------------------------------------------------
def test_list_pagination_and_order(client):
    for i in range(1, 6):
        emp(client, f"EMP000{i}")
        client.post("/attendance/punch-in", json={"emp_code": f"EMP000{i}", "punched_at": ms(2026, 7, 6 + i % 2, 9, 30)})
    j = client.get("/attendance?page=1&page_size=2").json()
    assert j["total"] == 5 and [(x["date"], x["emp_code"]) for x in j["items"]] == [("2026-07-07", "EMP0001"), ("2026-07-07", "EMP0003")]
    assert client.get("/attendance?emp_code=EMP0002").json()["total"] == 1
    e = client.get("/employees?page=2&page_size=2").json()
    assert e["total"] == 5 and [x["emp_code"] for x in e["items"]] == ["EMP0003", "EMP0004"]
    emp(client, "EMP0009", department="HR")
    assert client.get("/employees?department=HR").json()["total"] == 1


# ---- regularization ------------------------------------------------------------------------
def test_regularize(client):
    emp(client)
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 7, 10, 5)})
    client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 7, 19, 0)})
    url = "/attendance/EMP0001/2026-07-07"
    body = {"punch_in": ms(2026, 7, 7, 9, 28), "reason": "biometric glitch", "regularized_by": "hr.admin", "work_hours": 99}
    r = client.patch(url, json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["late_minutes"] == 0 and j["work_hours"] == 9.53 and len(j["history"]) == 1
    ch = j["history"][0]["changes"]
    assert set(ch) == {"punch_in", "late_minutes", "work_hours"}                  # only what actually changed
    assert ch["late_minutes"] == {"from": 35, "to": 0} and ch["punch_in"]["to"] == ms(2026, 7, 7, 9, 28)
    assert client.patch(url, json=body).status_code == 422                        # changes nothing
    j = client.patch(url, json={"status": "LEAVE", "reason": "approved leave", "regularized_by": "hr"}).json()
    assert j["punch_in"] is None and j["work_hours"] is None and len(j["history"]) == 2
    bad = {"status": "LEAVE", "punch_in": ms(2026, 7, 7, 9, 0), "reason": "approved leave", "regularized_by": "hr"}
    assert client.patch(url, json=bad).status_code == 422
    assert client.patch(url, json={"status": "WFH", "reason": "needs punch in", "regularized_by": "hr"}).status_code == 422
    wrong_day = {"status": "PRESENT", "punch_in": ms(2026, 7, 8, 9, 0), "reason": "wrong day!", "regularized_by": "hr"}
    assert client.patch(url, json=wrong_day).status_code == 422
    assert client.patch("/attendance/EMP0001/2026-07-30", json=body).status_code == 404
    assert client.patch("/attendance/NOPE/2026-07-07", json=body).status_code == 404
    assert client.patch("/attendance/EMP0001/not-a-date", json=body).status_code == 422


# ---- concurrency (same logic as with a real server: unique index + compare-and-set) ---------
def test_concurrent_punch_in_and_out(client):
    from concurrent.futures import ThreadPoolExecutor
    emp(client)
    body = {"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 9, 30)}
    with ThreadPoolExecutor(16) as ex:
        codes = list(ex.map(lambda _: client.post("/attendance/punch-in", json=body).status_code, range(16)))
    assert codes.count(201) == 1 and codes.count(409) == 15
    out = {"emp_code": "EMP0001", "punched_at": ms(2026, 7, 6, 18, 30)}
    with ThreadPoolExecutor(16) as ex:
        codes = list(ex.map(lambda _: client.post("/attendance/punch-out", json=out).status_code, range(16)))
    assert codes.count(200) == 1 and codes.count(409) == 15
