"""Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000."""

from airsense.api.main import create_app

app = create_app()
