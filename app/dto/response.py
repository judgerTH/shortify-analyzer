from pydantic import BaseModel
from typing import List, Optional


class DedupResponse(BaseModel):
    articleId: int
    isDuplicate: bool
    score: float
    model: Optional[str] = None


class DuplicateGroup(BaseModel):
    representativeId: int
    duplicateIds: List[int]
    groupSize: int
    maxSimilarity: float
    model: Optional[str] = None


class GroupDedupResponse(BaseModel):
    model: str
    threshold: float
    totalArticles: int
    uniqueGroups: int
    duplicatesRemoved: int
    costReductionRate: float
    representativeIds: List[int]
    groups: List[DuplicateGroup]


class ModelComparisonResult(BaseModel):
    uniqueGroups: int
    duplicatesRemoved: int
    costReductionRate: float
    threshold: float
    groups: List[DuplicateGroup]


class CompareResponse(BaseModel):
    total: int
    tfidf: ModelComparisonResult
    sbert: ModelComparisonResult
