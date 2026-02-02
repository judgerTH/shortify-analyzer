from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.dedup import router as dedup_router

app = FastAPI(title="Shortify Analyzer")

app.include_router(health_router)
app.include_router(dedup_router)
