from pydantic import BaseModel

class DedupRequest(BaseModel):
    articleId: int
    title: str
    content: str
