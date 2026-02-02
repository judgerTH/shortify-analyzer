from pydantic import BaseModel

class DedupResponse(BaseModel):
    articleId: int
    isDuplicate: bool
    score: float
