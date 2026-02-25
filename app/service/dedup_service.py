from typing import List, Dict, Any, Optional
from app.core.config import DedupConfig
from app.service.tfidf_service import TfidfDedupService
from app.service.sbert_service import SbertDedupService


class DedupService:
    """
    중복 판별 통합 서비스.
    - model="tfidf" | "sbert" 선택 가능
    - group_duplicates: 한 배치의 기사들을 그룹화 → LLM 호출 절감
    - check_batch: 개별 중복 여부 반환 (기존 API 호환)
    - compare_models: 두 모델 결과 동시 반환 (실험/비교용)
    """

    def __init__(self):
        self._tfidf = TfidfDedupService()
        self._sbert: Optional[SbertDedupService] = None

    def _get_sbert(self) -> SbertDedupService:
        if self._sbert is None:
            self._sbert = SbertDedupService()
        return self._sbert

    def _get_service(self, model: str):
        if model == "sbert":
            return self._get_sbert()
        return self._tfidf

    def check_batch(self, requests) -> List[Dict[str, Any]]:
        """기존 API 호환 - isDuplicate / score 반환"""
        articles = [
            {"id": r.articleId, "title": r.title, "content": r.content}
            for r in requests
        ]
        model = DedupConfig.DEFAULT_MODEL
        return self._get_service(model).check_batch(articles)

    def group_duplicates(
        self,
        articles: List[Dict[str, Any]],
        model: str = None,
    ) -> Dict[str, Any]:
        """
        기사 배치를 중복 그룹으로 묶고, LLM에 보낼 대표 기사 ID 리스트 반환.
        비용 절감 측정 포함.

        반환 예시:
        {
            "model": "tfidf",
            "threshold": 0.85,
            "totalArticles": 20,
            "uniqueGroups": 14,
            "duplicatesRemoved": 6,
            "costReductionRate": 0.30,
            "representativeIds": [1, 3, 5, ...],
            "groups": [...]
        }
        """
        model = model or DedupConfig.DEFAULT_MODEL
        svc = self._get_service(model)
        groups = svc.group_duplicates(articles)

        total = len(articles)
        unique = len(groups)
        removed = total - unique
        reduction_rate = round(removed / total, 4) if total > 0 else 0.0

        representative_ids = [g["representativeId"] for g in groups]
        threshold = (
            DedupConfig.SBERT_THRESHOLD if model == "sbert"
            else DedupConfig.TFIDF_THRESHOLD
        )

        return {
            "model": model,
            "threshold": threshold,
            "totalArticles": total,
            "uniqueGroups": unique,
            "duplicatesRemoved": removed,
            "costReductionRate": reduction_rate,
            "representativeIds": representative_ids,
            "groups": groups,
        }

    def compare_models(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        TF-IDF vs SBERT 결과 비교 반환.
        포트폴리오용 실험 엔드포인트.
        """
        tfidf_result = self._tfidf.group_duplicates(articles)
        sbert_result = self._get_sbert().group_duplicates(articles)

        tfidf_unique = len(tfidf_result)
        sbert_unique = len(sbert_result)
        total = len(articles)

        return {
            "total": total,
            "tfidf": {
                "uniqueGroups": tfidf_unique,
                "duplicatesRemoved": total - tfidf_unique,
                "costReductionRate": round((total - tfidf_unique) / total, 4) if total else 0,
                "threshold": DedupConfig.TFIDF_THRESHOLD,
                "groups": tfidf_result,
            },
            "sbert": {
                "uniqueGroups": sbert_unique,
                "duplicatesRemoved": total - sbert_unique,
                "costReductionRate": round((total - sbert_unique) / total, 4) if total else 0,
                "threshold": DedupConfig.SBERT_THRESHOLD,
                "groups": sbert_result,
            },
        }
