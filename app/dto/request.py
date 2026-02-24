from pydantic import BaseModel
from typing import Optional, List


class DedupRequest(BaseModel):
    articleId: int
    title: str
    content: str


class GroupDedupRequest(BaseModel):
    articles: List[DedupRequest]
    model: Optional[str] = None  # "tfidf" | "sbert" | None(=config default)


class CompareRequest(BaseModel):
    articles: List[DedupRequest]
