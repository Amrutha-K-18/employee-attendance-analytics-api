# Employee Attendance & Analytics API

FastAPI + MongoDB. All application code is in `app/main.py`.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MONGO_URI=mongodb://localhost:27017 MONGO_DB=attendance_db   # or copy .env.example to .env
uvicorn app.main:app --port 8000
```
MongoDB 6.0+ is required (`$setWindowFields`). Indexes are created at startup, nothing else to do.
Local MongoDB: `docker run -d -p 27017:27017 mongo:7`.

## Tests
```bash
pip install pytest mongomock httpx
pytest tests/test_rules.py                      # no database needed (mongomock)
MONGO_DB=attendance_test pytest tests/test_analytics_live.py   # needs a real MongoDB; DROPS that database
```

## Notes
- Everything in the contract is implemented, including the explain endpoint.
- Interpretation choices where the contract is open: the trend range limit is 92 calendar days *inclusive* (93 days -> 422);
  `moving_avg_7d` is the mean of the *rounded* (4 dp) daily rates, rounded half-up again; a request to `PATCH` that
  loses an optimistic-concurrency race returns 409.
- Rounding is half-up everywhere (done with exact decimal / integer arithmetic, not `$round`, which is banker's).
