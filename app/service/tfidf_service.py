import re
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

from app.core.config import DedupConfig

# 뉴스 boilerplate 제거 패턴 (제보 안내, 이메일, URL 등)
_BOILERPLATE_RE = re.compile(
    r"(※.{0,100}|▶.{0,100}|▷.{0,100}|"
    r"이메일\s*:\s*\S+|카카오톡\s*:\s*\S+|"
    r"[가-힣a-zA-Z]+@[a-zA-Z0-9.]+\.[a-zA-Z]{2,}|"
    r"https?://\S+|무단.{0,20}금지|저작권.{0,30}|"
    r"\[\s*사진\s*=.{0,50}\]|\(\s*[가-힣]+\s*제공\s*\))",
    re.MULTILINE,
)
# 속보/단독 등 prefix 제거
_PREFIX_RE = re.compile(r"^\s*\[(속보|단독|긴급|종합|업데이트)\]\s*")


def _parse_date(s: Any) -> Optional[datetime]:
    if not s or str(s) in ("None", ""):
        return None
    try:
        return datetime.fromisoformat(str(s).split(".")[0])
    except Exception:
        return None


class TfidfDedupService:
    """
    TF-IDF + Cosine Similarity 기반 중복 판별.
    - 빠르고 가벼움
    - 형태소 분석 없이도 한국어 char-ngram으로 동작
    - title_weight: 제목을 N배 반복하여 본문보다 강조
    - date_window_days: 발행일 ±N일 내 기사끼리만 비교
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=DedupConfig.TFIDF_MAX_FEATURES,
            ngram_range=DedupConfig.TFIDF_NGRAM_RANGE,
            analyzer="char_wb",
            min_df=1,
        )
        self.threshold        = DedupConfig.TFIDF_THRESHOLD
        self.title_weight     = DedupConfig.TITLE_WEIGHT
        self.date_window_days = DedupConfig.DATE_WINDOW_DAYS

    def _preprocess(self, title: str, content: str) -> str:
        """
        - boilerplate 제거 후 본문 앞부분 사용
        - 제목을 title_weight배 반복하여 의미적 가중치 부여
        """
        title_norm = _PREFIX_RE.sub("", title).strip()
        content_clean = _BOILERPLATE_RE.sub(" ", content[: DedupConfig.CONTENT_MAX_CHARS * 2])
        content_clean = content_clean[: DedupConfig.CONTENT_MAX_CHARS].strip()

        if self.title_weight > 1:
            return (title_norm + " ") * self.title_weight + content_clean
        return title_norm + " " + content_clean

    def group_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        articles: [{"id": int, "title": str, "content": str, "published_at": str(optional)}, ...]
        반환: [{"representativeId": int, "duplicateIds": [int], "groupSize": int, "maxSimilarity": float}, ...]

        Union-Find로 중복 그룹 묶기.
        date_window_days > 0 이면 발행일 ±N일 내 기사끼리만 비교.
        """
        if not articles:
            return []

        texts  = [self._preprocess(a["title"], a["content"]) for a in articles]
        ids    = [a["id"] for a in articles]
        dates  = [_parse_date(a.get("published_at")) for a in articles]
        n      = len(texts)
        window = timedelta(days=self.date_window_days) if self.date_window_days > 0 else None

        vectors    = self.vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(vectors)

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for i in range(n):
            for j in range(i + 1, n):
                # 날짜 필터
                if window and dates[i] and dates[j]:
                    if abs(dates[i] - dates[j]) > window:
                        continue
                if sim_matrix[i][j] >= self.threshold:
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        result = []
        for root_idx, member_idxs in groups.items():
            if len(member_idxs) == 1:
                representative_idx = member_idxs[0]
                max_sim = 0.0
            else:
                avg_sims = [
                    np.mean([sim_matrix[idx][m] for m in member_idxs if m != idx])
                    for idx in member_idxs
                ]
                representative_idx = member_idxs[int(np.argmax(avg_sims))]
                max_sim = float(np.max(avg_sims))

            result.append({
                "representativeId": ids[representative_idx],
                "duplicateIds":     [ids[m] for m in member_idxs if m != representative_idx],
                "groupSize":        len(member_idxs),
                "maxSimilarity":    round(max_sim, 4),
                "model":            "tfidf",
            })

        return result

    def check_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        실시간 스트리밍용: 순차 처리로 isDuplicate 플래그 반환.
        ※ group_duplicates와 달리 전체 IDF가 아닌 누적 IDF를 사용하므로
          유사도 수치가 다를 수 있음 (설계상 trade-off).
        """
        if not articles:
            return []

        texts   = [self._preprocess(a["title"], a["content"]) for a in articles]
        results = []
        accepted_texts = []

        for i, article in enumerate(articles):
            text = texts[i]

            if not accepted_texts:
                accepted_texts.append(text)
                results.append({
                    "articleId":   article["id"],
                    "isDuplicate": False,
                    "score":       0.0,
                    "model":       "tfidf",
                })
                continue

            all_texts = accepted_texts + [text]
            vectors   = self.vectorizer.fit_transform(all_texts)
            sims      = cosine_similarity(vectors[-1], vectors[:-1])[0]
            max_score = float(sims.max())
            is_dup    = max_score >= self.threshold

            results.append({
                "articleId":   article["id"],
                "isDuplicate": is_dup,
                "score":       round(max_score, 4),
                "model":       "tfidf",
            })

            if not is_dup:
                accepted_texts.append(text)

        return results
