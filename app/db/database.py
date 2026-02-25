from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:password@localhost:3306/shortify"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def fetch_articles_for_dedup(db, limit: int = 500):
    """
    COLLECTED 상태의 original_article을 가져옴
    article_meta와 조인하여 아직 중복 판별 안 된 기사만 반환
    """
    query = text("""
        SELECT oa.id, oa.title, oa.content, oa.url
        FROM shortify.original_article oa
        INNER JOIN shortify.article_meta am ON am.url = oa.url
        WHERE am.status = 'COLLECTED'
        ORDER BY oa.crawled_at DESC
        LIMIT :limit
    """)
    rows = db.execute(query, {"limit": limit}).fetchall()
    return [{"id": row.id, "title": row.title, "content": row.content, "url": row.url} for row in rows]
