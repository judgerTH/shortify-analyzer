import numpy as np
from typing import List, Dict, Any

from app.core.config import DedupConfig


class SbertDedupService:
    """
    Sentence-BERT 기반 중복 판별.
    - 의미론적 유사도 측정 가능 (TF-IDF보다 정확)
    - 무겁지만 더 좋은 품질
    - 포트폴리오에서 TF-IDF 대비 비용/성능 비교 실험용
    """

    def __init__(self):
        self.threshold = DedupConfig.SBERT_THRESHOLD
        self._model = None  # lazy load

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(DedupConfig.SBERT_MODEL_NAME)
        return self._model

    def _preprocess(self, title: str, content: str) -> str:
        combined = title + " " + content[:DedupConfig.CONTENT_MAX_CHARS]
        return combined.strip()

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def group_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not articles:
            return []

        model = self._get_model()
        texts = [self._preprocess(a["title"], a["content"]) for a in articles]
        ids = [a["id"] for a in articles]
        n = len(texts)

        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # cosine similarity = dot product (embeddings이 normalized)
        sim_matrix = np.dot(embeddings, embeddings.T)

        # Union-Find
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
                if sim_matrix[i][j] >= self.threshold:
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        result = []
        for root_idx, member_idxs in groups.items():
            if len(member_idxs) == 1:
                representative_idx = member_idxs[0]
                max_sim = 0.0
            else:
                avg_sims = []
                for idx in member_idxs:
                    others = [m for m in member_idxs if m != idx]
                    avg_sim = float(np.mean([sim_matrix[idx][o] for o in others]))
                    avg_sims.append(avg_sim)
                representative_idx = member_idxs[int(np.argmax(avg_sims))]
                max_sim = float(np.max(avg_sims))

            duplicate_ids = [ids[m] for m in member_idxs if m != representative_idx]

            result.append({
                "representativeId": ids[representative_idx],
                "duplicateIds": duplicate_ids,
                "groupSize": len(member_idxs),
                "maxSimilarity": round(max_sim, 4),
                "model": "sbert",
            })

        return result

    def check_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not articles:
            return []

        model = self._get_model()
        texts = [self._preprocess(a["title"], a["content"]) for a in articles]
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        results = []
        accepted_embeddings = []
        accepted_ids = []

        for i, article in enumerate(articles):
            emb = embeddings[i]

            if not accepted_embeddings:
                accepted_embeddings.append(emb)
                accepted_ids.append(article["id"])
                results.append({
                    "articleId": article["id"],
                    "isDuplicate": False,
                    "score": 0.0,
                    "model": "sbert",
                })
                continue

            sims = np.dot(np.array(accepted_embeddings), emb)
            max_score = float(sims.max())
            is_dup = max_score >= self.threshold

            results.append({
                "articleId": article["id"],
                "isDuplicate": is_dup,
                "score": round(max_score, 4),
                "model": "sbert",
            })

            if not is_dup:
                accepted_embeddings.append(emb)
                accepted_ids.append(article["id"])

        return results
