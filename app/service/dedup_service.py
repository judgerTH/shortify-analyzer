from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DedupService:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.corpus = []          # 기존 기사 텍스트
        self.article_ids = []     # 기존 기사 ID

    def check_batch(self, requests):
        texts = [r.title + " " + r.content for r in requests]

        # 최초 요청
        if not self.corpus:
            self.corpus.extend(texts)
            self.article_ids.extend([r.articleId for r in requests])
            return [
                {
                    "articleId": r.articleId,
                    "isDuplicate": False,
                    "score": 0.0
                }
                for r in requests
            ]

        all_texts = self.corpus + texts
        vectors = self.vectorizer.fit_transform(all_texts)

        old_vectors = vectors[:len(self.corpus)]
        new_vectors = vectors[len(self.corpus):]

        results = []

        for idx, vec in enumerate(new_vectors):
            sims = cosine_similarity(vec, old_vectors)[0]
            max_score = float(sims.max())

            is_dup = max_score >= 0.85

            results.append({
                "articleId": requests[idx].articleId,
                "isDuplicate": is_dup,
                "score": round(max_score, 4)
            })

            if not is_dup:
                self.corpus.append(texts[idx])
                self.article_ids.append(requests[idx].articleId)

        return results
