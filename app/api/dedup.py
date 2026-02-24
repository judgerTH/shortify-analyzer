from fastapi import APIRouter
from typing import List

from app.dto.request import DedupRequest, GroupDedupRequest, CompareRequest
from app.dto.response import DedupResponse, GroupDedupResponse, CompareResponse
from app.service.dedup_service import DedupService

router = APIRouter(prefix="/analysis/dedup", tags=["dedup"])
service = DedupService()


@router.post("/batch", response_model=List[DedupResponse])
def dedup_batch(requests: List[DedupRequest]):
    """
    [기존 API 호환] 기사 리스트의 순차적 중복 여부 판별.
    각 기사가 이전 기사들과 중복인지 체크.
    """
    return service.check_batch(requests)


@router.post("/group", response_model=GroupDedupResponse)
def dedup_group(request: GroupDedupRequest):
    """
    기사 배치를 중복 그룹으로 묶어 대표 기사 선별.
    LLM 요약 호출 전에 사용 → 비용 절감.

    - representativeIds: LLM에 넘길 기사 ID 목록
    - costReductionRate: 절감 비율 (0.0 ~ 1.0)
    - model: "tfidf" | "sbert" (미입력시 환경변수 기본값)
    """
    articles = [
        {"id": r.articleId, "title": r.title, "content": r.content}
        for r in request.articles
    ]
    return service.group_duplicates(articles, model=request.model)


@router.post("/compare", response_model=CompareResponse)
def dedup_compare(request: CompareRequest):
    """
    TF-IDF vs SBERT 중복 판별 결과 비교.
    모델별 비용 절감 효과 측정용 (포트폴리오 실험 엔드포인트).
    """
    articles = [
        {"id": r.articleId, "title": r.title, "content": r.content}
        for r in request.articles
    ]
    return service.compare_models(articles)
