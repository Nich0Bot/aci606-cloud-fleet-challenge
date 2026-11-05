# API contract

The evaluator assumes these routes exist:

- `GET /health`
- `POST /telemetry`
- `GET /robots`
- `GET /robots/{robot_id}/history`
- `POST /predict`
- `GET /metrics`
- `GET /dashboard`

Do not rename the required JSON fields used by `/predict` without instructor approval.
