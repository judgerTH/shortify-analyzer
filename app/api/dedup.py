from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/analysis", tags=["analysis"])


class DedupRequest(BaseModel):
    title: str
    content: str


class DedupResponse(BaseModel):
    is_duplicate: bool
    score: float


@router.post("/dedup", response_model=DedupResponse)
def dedup(req: DedupRequest):
    # 지금은 연결 확인용 더미 구현
    return DedupResponse(
        is_duplicate=False,
        score=0.0
    )
