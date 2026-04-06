# Deployment Guide for Interview Demo

This guide gives you two deployment tracks:
- Track A (fastest demo): public URL in under 20 minutes.
- Track B (production-style): AWS stack with managed Redis and S3 model storage.

## Quick Start: GitHub + Render (Recommended)

This project already serves both API and UI from the same FastAPI app, so you do not need Vercel unless you want a separate frontend deployment.

### 1) Push to GitHub
1. Create a new GitHub repo.
2. From the project root, run:

```bash
git init
git add .
git commit -m "Initial fraud engine deployment"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

### 2) Deploy on Render using Blueprint
1. In Render dashboard: New > Blueprint.
2. Select your GitHub repository.
3. Render detects `render.yaml` and provisions:
   - `fraud-engine-api` (web service)
4. Deploy.

Notes:
- Build step generates synthetic data and trains a model artifact during deploy, so deployment does not depend on pre-committed data/model files.
- Redis is optional. If `REDIS_URL` is not provided, the app automatically falls back to in-memory state store.
- API key is optional for quick demos (default `local-dev-key`), but set `API_KEY` in production.

### 3) Validate deployment
1. Open `/health` and verify `status=ok`.
2. Open `/` and run a prediction from the UI.
3. Test `/predict` with `x-api-key`.
4. Open `/metrics` and verify counters/histogram.

### 4) Optional: Vercel (only if you want separate frontend hosting)
You can host only a static UI on Vercel, but then you must wire API requests to your Render backend and handle CORS/route proxying. For interviews, same-origin UI from Render is simpler and more reliable.

## Track A: Fastest Public Demo (Render + Upstash)

### Why this is good for interviews
- Very fast to publish a working URL.
- No local laptop dependency during interview.
- Shows cloud + managed state store usage.

### Steps
1. Push this repository to GitHub.
2. Create an Upstash Redis database and copy the rediss:// connection URL.
3. In Render, create a new Web Service from this repository.
   - Optional: use the render blueprint in render.yaml.
4. Configure build/start:
   - Build command: pip install -r requirements.txt
   - Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
5. Set environment variables in Render:
   - API_KEY=<strong random value>
   - REDIS_URL=<your Upstash rediss URL>
   - REDIS_WINDOW_SIZE=5
   - MODEL_PATH=artifacts/fraud_model.pkl
   - MODEL_VERSION=xgboost-v1
6. Deploy and validate:
   - GET /health should return status ok or degraded with model_loaded true.
   - POST /predict with x-api-key should return score + features.

### Interview Script
- Open /health first.
- Send a few /predict requests with same user_id to show changing rolling features.
- Open /metrics and point out prediction counters and latency histogram.

## Track B: Production-Style AWS (App Runner + ElastiCache + S3)

### Architecture
- API: AWS App Runner service from ECR image.
- Redis: ElastiCache for Redis.
- Model artifact: S3, loaded with MODEL_S3_URI on startup.

### Steps
1. Create ECR repository and push Docker image.
2. Upload model artifact to S3, for example s3://fraud-engine-models/fraud_model.pkl.
3. Create ElastiCache Redis and obtain endpoint/port.
4. Create App Runner service from ECR image.
5. Set environment variables:
   - API_KEY=<strong random value>
   - REDIS_URL=redis://<elasticache-endpoint>:6379/0
   - MODEL_S3_URI=s3://fraud-engine-models/fraud_model.pkl
   - MODEL_VERSION=xgboost-v1
   - REDIS_WINDOW_SIZE=5
6. Attach IAM role to App Runner with s3:GetObject permission for the model bucket.
7. Validate endpoints and run remote load test against App Runner URL.

## What to Show Interviewers
1. Live API URL and health check.
2. A short screen recording of load test execution.
3. Raw benchmark artifacts from artifacts/benchmarks.
4. CI pipeline run status from GitHub Actions.

## Security Notes for Demo
- Keep API_KEY private and rotate after demo.
- Restrict CORS to your frontend domain if you expose a UI.
- Do not store plain PII in Redis; hash/tokenize user identifiers.
