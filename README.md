# agent-system

A mini agent registry + usage tracking API built with FastAPI.

## setup

```bash
pip install fastapi uvicorn
uvicorn main:app --reload
```

visit http://127.0.0.1:8000/docs to test all endpoints interactively.

---

## design questions

### 1. billing without double charging?

The `request_id` idempotency check already handles duplicate usage events at the logging level. For billing I'd persist each `request_id` in a DB with a `billed` boolean. Before charging, query only unbilled entries, process them, then flip the flag in the same transaction. That way even if the same event comes in twice, it only gets billed once.

### 2. storage at 100K agents?

In-memory breaks down pretty fast. I'd move to SQLite first (good enough up to maybe 10K agents), then PostgreSQL for anything larger. The search endpoint would need a proper full-text index on `name` and `description` — otherwise it's just a full table scan which won't scale. Usage logs would go into a separate table with a foreign key to agents so I can do `GROUP BY target` instead of looping in Python.
