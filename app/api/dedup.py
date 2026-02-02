from fastapi import APIRouter
from typing import List

from app.dto.request import DedupRequest
from app.dto.response import DedupResponse
from app.service.dedup_service import DedupService

router = APIRouter()
service = DedupService()

@router.post("/analysis/dedup/batch", response_model=List[DedupResponse])
def dedup_batch(requests: List[DedupRequest]):
    return service.check_batch(requests)
