import os
from dotenv import load_dotenv

load_dotenv()


class DedupConfig:
    # 중복 판정 threshold (cosine similarity 기준)
    TFIDF_THRESHOLD: float = float(os.getenv("TFIDF_THRESHOLD", "0.85"))
    SBERT_THRESHOLD: float = float(os.getenv("SBERT_THRESHOLD", "0.88"))

    # 사용할 모델 ("tfidf" | "sbert")
    DEFAULT_MODEL: str = os.getenv("DEDUP_MODEL", "tfidf")

    # SBERT 모델명 (한국어 지원 모델)
    SBERT_MODEL_NAME: str = os.getenv(
        "SBERT_MODEL_NAME",
        "jhgan/ko-sroberta-multitask"
    )

    # TF-IDF 설정
    TFIDF_MAX_FEATURES: int = int(os.getenv("TFIDF_MAX_FEATURES", "10000"))
    TFIDF_NGRAM_RANGE: tuple = (1, 2)

    # 기사 본문 최대 길이 (토큰 절약)
    CONTENT_MAX_CHARS: int = int(os.getenv("CONTENT_MAX_CHARS", "500"))

    # 제목 가중치: 제목을 N배 반복하여 본문보다 강조 (0=비활성화)
    TITLE_WEIGHT: int = int(os.getenv("TITLE_WEIGHT", "3"))

    # 날짜 필터: 발행일 기준 ±N일 내 기사끼리만 비교 (0=비활성화)
    DATE_WINDOW_DAYS: int = int(os.getenv("DATE_WINDOW_DAYS", "1"))
