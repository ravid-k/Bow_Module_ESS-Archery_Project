# ESS-ARCHERY Vision API

Render-ready FastAPI wrapper around ESS-ARCHERY Vision V0.5.

## Render settings
- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn vision_api:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

After deployment, open `https://YOUR-SERVICE.onrender.com/health`.
Use only the base URL (`https://YOUR-SERVICE.onrender.com`) in the ESS-ARCHERY dashboard Vision API URL field.
