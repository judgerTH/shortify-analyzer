import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

from app.core.config import DedupConfig


class TfidfDedupService:
    """
    TF-IDF + Cosine Similarity 기반 중복 판별.
    - 빠르고 가벼움
    - 형태소 분석 없이도 한국어 ngram으로 어느정도 동작
    - 기준: threshold 이상이면 중복
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=DedupConfig.TFIDF_MAX_FEATURES,
            ngram_range=DedupConfig.TFIDF_NGRAM_RANGE,
            analyzer="char_wb",  # 한국어에서 char-level ngram이 더 효과적
            min_df=1,
        )
        self.threshold = DedupConfig.TFIDF_THRESHOLD

    def _preprocess(self, title: str, content: str) -> str:
        """제목 + 본문 앞부분만 사용 (노이즈 줄이기)"""
        combined = title + " " + content[:DedupConfig.CONTENT_MAX_CHARS]
        return combined.strip()

    def group_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        articles: [{"id": int, "title": str, "content": str}, ...]
        반환: [{"representativeId": int, "duplicateIds": [int], "groupSize": int}, ...]
        
        Union-Find로 중복 그룹 묶기
        """
        if not articles:
            return []

        texts = [self._preprocess(a["title"], a["content"]) for a in articles]
        ids = [a["id"] for a in articles]
        n = len(texts)

        # TF-IDF 벡터화
        vectors = self.vectorizer.fit_transform(texts)

        # Union-Find
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        # 유사도 계산 (배치로 한 번에)
        sim_matrix = cosine_similarity(vectors)

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= self.threshold:
                    union(i, j)

        # 그룹 묶기
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        result = []
        for root_idx, member_idxs in groups.items():
            # 대표 기사: 그룹 내에서 다른 기사들과 평균 유사도가 가장 높은 것
            if len(member_idxs) == 1:
                representative_idx = member_idxs[0]
                max_sim = 0.0
            else:
                avg_sims = []
                for idx in member_idxs:
                    others = [m for m in member_idxs if m != idx]
                    avg_sim = np.mean([sim_matrix[idx][o] for o in others])
                    avg_sims.append(avg_sim)
                representative_idx = member_idxs[int(np.argmax(avg_sims))]
                max_sim = float(np.max(avg_sims))

            duplicate_ids = [ids[m] for m in member_idxs if m != representative_idx]

            result.append({
                "representativeId": ids[representative_idx],
                "duplicateIds": duplicate_ids,
                "groupSize": len(member_idxs),
                "maxSimilarity": round(max_sim, 4),
                "model": "tfidf",
            })

        return result

    def check_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        개별 기사 리스트에 대해 isDuplicate 플래그 반환.
        순차적으로 처리: 이미 처리한 기사 대비 중복 여부 판별.
        """
        if not articles:
            return []

        texts = [self._preprocess(a["title"], a["content"]) for a in articles]
        results = []
        accepted_texts = []
        accepted_ids = []

        for i, article in enumerate(articles):
            text = texts[i]

            if not accepted_texts:
                accepted_texts.append(text)
                accepted_ids.append(article["id"])
                results.append({
                    "articleId": article["id"],
                    "isDuplicate": False,
                    "score": 0.0,
                    "model": "tfidf",
                })
                continue

            all_texts = accepted_texts + [text]
            vectors = self.vectorizer.fit_transform(all_texts)
            new_vec = vectors[-1]
            old_vecs = vectors[:-1]

            sims = cosine_similarity(new_vec, old_vecs)[0]
            max_score = float(sims.max())
            is_dup = max_score >= self.threshold

            results.append({
                "articleId": article["id"],
                "isDuplicate": is_dup,
                "score": round(max_score, 4),
                "model": "tfidf",
            })

            if not is_dup:
                accepted_texts.append(text)
                accepted_ids.append(article["id"])

        return results
