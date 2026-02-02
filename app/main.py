from fastapi import FastAPI
from app.api.dedup import router as dedup_router

app = FastAPI()
app.include_router(dedup_router)
