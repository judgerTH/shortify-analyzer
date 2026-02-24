from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.dedup import router as dedup_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 TF-IDF 워밍업 (SBERT는 lazy load)
    print("[Analyzer] Starting up...")
    yield
    print("[Analyzer] Shutting down...")


app = FastAPI(
    title="Shortify Analyzer",
    description="뉴스 기사 중복 판별 서비스 (TF-IDF / Sentence-BERT)",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(dedup_router)

