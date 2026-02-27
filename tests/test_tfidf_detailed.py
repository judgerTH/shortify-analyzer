"""
TF-IDF 중복 기사 판별 - 정밀 테스트
테스트 케이스:
  1. 완전 동일 기사 (copy)
  2. 약간 변형된 중복 (앞부분 추가/삭제)
  3. 같은 사건 다른 언론사 (어조 다름)
  4. 완전 다른 토픽
  5. 임계값 경계 케이스
  6. 빈 리스트 / 단건
  7. 그룹화 정확성 (Union-Find)
  8. 대표 기사 선정 정확성
  9. check_batch vs group_duplicates 일관성
"""
import sys
sys.path.insert(0, "/home/claude/shortify-analyzer")

import json
import time
import numpy as np
from app.service.tfidf_service import TfidfDedupService
from app.core.config import DedupConfig

svc = TfidfDedupService()

results = []

def record(name, passed, detail="", similarity=None):
    icon = "✅" if passed else "❌"
    results.append({
        "name": name,
        "passed": passed,
        "detail": detail,
        "similarity": similarity,
    })
    sim_str = f" | similarity={similarity:.4f}" if similarity is not None else ""
    print(f"{icon} {name}{sim_str}")
    if detail:
        print(f"   → {detail}")

# ──────────────────────────────────────────────
# 케이스 1: 완전 동일 기사
# ──────────────────────────────────────────────
articles_identical = [
    {"id": 1, "title": "삼성전자 3분기 영업이익 10조원 달성", "content": "삼성전자가 3분기 영업이익 10조원을 달성했다고 발표했다. 반도체 부문의 실적 회복이 주효했으며, HBM 수요 증가가 주요 원인으로 분석된다."},
    {"id": 2, "title": "삼성전자 3분기 영업이익 10조원 달성", "content": "삼성전자가 3분기 영업이익 10조원을 달성했다고 발표했다. 반도체 부문의 실적 회복이 주효했으며, HBM 수요 증가가 주요 원인으로 분석된다."},
]
groups = svc.group_duplicates(articles_identical)
passed = len(groups) == 1 and groups[0]["groupSize"] == 2
sim = groups[0]["maxSimilarity"] if groups else 0
record("완전 동일 기사 → 1그룹", passed, f"groups={len(groups)}, groupSize={groups[0]['groupSize'] if groups else 0}", sim)

# ──────────────────────────────────────────────
# 케이스 2: 약간 변형 (앞문장 추가)
# ──────────────────────────────────────────────
articles_minor_diff = [
    {"id": 1, "title": "삼성전자 반도체 실적 개선", "content": "삼성전자 반도체 부문이 HBM 수요 급증으로 인해 올해 3분기 큰 폭의 실적 개선을 이뤄냈다. 영업이익은 전분기 대비 30% 상승했다."},
    {"id": 2, "title": "삼성전자 반도체 실적 개선 소식", "content": "[속보] 삼성전자 반도체 부문이 HBM 수요 급증으로 인해 올해 3분기 큰 폭의 실적 개선을 이뤄냈다. 영업이익은 전분기 대비 30% 상승했다."},
]
groups = svc.group_duplicates(articles_minor_diff)
sim = groups[0]["maxSimilarity"] if len(groups) == 1 else 0
passed = len(groups) == 1
record("약간 변형 기사 → 중복 탐지", passed, f"groups={len(groups)}", sim)

# ──────────────────────────────────────────────
# 케이스 3: 같은 사건, 다른 언론사 (어조/표현 다름)
# ──────────────────────────────────────────────
articles_same_event = [
    {"id": 1, "title": "삼성전자, 3Q 영업익 10조 돌파", "content": "삼성전자가 3분기 영업이익 10조원을 넘어섰다. 메모리 반도체 수요 회복과 HBM 공급 확대가 실적을 견인했다. 전문가들은 4분기도 긍정적으로 전망했다."},
    {"id": 2, "title": "삼성전자 3분기 실적, 영업이익 10조원 상회", "content": "삼성전자의 3분기 영업이익이 시장 예상을 웃도는 10조원을 기록했다. HBM 등 고부가 메모리 제품의 수요가 확대된 결과로 분석된다. 증권가는 긍정적 전망을 유지했다."},
]
groups = svc.group_duplicates(articles_same_event)
sim = groups[0]["maxSimilarity"] if len(groups) == 1 else None
passed_dup = len(groups) == 1
# 이 케이스는 탐지 안 될 수도 있음 - 실제 sim 값이 중요
batch = svc.check_batch(articles_same_event)
batch_sim = batch[1]["score"] if len(batch) > 1 else 0
record(
    "같은 사건 다른 언론사 유사도 측정",
    True,  # 통과/실패보다 수치 확인이 목적
    f"group_duplicates: {len(groups)}그룹, sim={sim if sim else '분리됨'} | check_batch score={batch_sim:.4f}",
    batch_sim
)

# ──────────────────────────────────────────────
# 케이스 4: 완전히 다른 토픽
# ──────────────────────────────────────────────
articles_diff_topic = [
    {"id": 1, "title": "코스피 2600선 돌파", "content": "코스피 지수가 외국인 매수세에 힘입어 2600선을 돌파했다. 반도체, 자동차 업종이 주도하며 상승 마감했다."},
    {"id": 2, "title": "태풍 북상 중부지방 영향 예고", "content": "기상청은 태풍이 빠른 속도로 북상하며 이번 주말 중부지방에 직접 영향을 줄 것이라고 예보했다. 주민들의 안전 대비가 요구된다."},
    {"id": 3, "title": "삼성전자 갤럭시 신제품 발표", "content": "삼성전자가 신형 갤럭시 시리즈를 공개했다. 온디바이스 AI와 카메라 성능이 대폭 강화됐으며 가격은 전작과 유사하다."},
]
groups = svc.group_duplicates(articles_diff_topic)
passed = len(groups) == 3
record("완전히 다른 토픽 3개 → 3그룹 분리", passed, f"groups={len(groups)}")

# ──────────────────────────────────────────────
# 케이스 5: 임계값 경계 (threshold 근처)
# ──────────────────────────────────────────────
# 유사하지만 다른 내용 - threshold 변화에 따른 결과 차이 확인
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text_a = "삼성전자 반도체 부문 실적이 3분기에 대폭 개선됐다. HBM 수요 증가 영향"
text_b = "삼성전자 반도체 사업 성과가 3분기에 크게 좋아졌다. HBM 수요 확대 원인"

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1,2), min_df=1)
m = vec.fit_transform([text_a, text_b])
raw_sim = float(cosine_similarity(m[0], m[1])[0][0])

# threshold=0.7로 낮추면 중복 탐지되는지
import os
os.environ["TFIDF_THRESHOLD"] = "0.7"
from importlib import reload
import app.core.config as conf
reload(conf)
from app.service import tfidf_service as tfidf_mod
reload(tfidf_mod)
svc_low = tfidf_mod.TfidfDedupService()
svc_low.threshold = 0.7

arts = [
    {"id": 10, "title": "삼성전자 반도체 부문 실적 개선", "content": text_a},
    {"id": 11, "title": "삼성전자 반도체 사업 성과 개선", "content": text_b},
]
groups_low = svc_low.group_duplicates(arts)
groups_orig = svc.group_duplicates(arts)  # threshold 0.85
record(
    f"임계값 경계 케이스 (raw_sim={raw_sim:.4f})",
    True,
    f"threshold=0.85 → {len(groups_orig)}그룹 | threshold=0.70 → {len(groups_low)}그룹",
    raw_sim
)

# ──────────────────────────────────────────────
# 케이스 6: 엣지 케이스 (빈 리스트, 단건)
# ──────────────────────────────────────────────
empty = svc.group_duplicates([])
record("빈 리스트 → 빈 결과", empty == [], f"result={empty}")

single = svc.group_duplicates([{"id": 99, "title": "단건 기사", "content": "내용"}])
passed = len(single) == 1 and single[0]["groupSize"] == 1 and single[0]["representativeId"] == 99
record("단건 기사 → 1그룹, 자기 자신이 대표", passed, f"result={single}")

# ──────────────────────────────────────────────
# 케이스 7: 그룹화 정확성 (3개 중 2개가 중복)
# ──────────────────────────────────────────────
articles_3way = [
    {"id": 1, "title": "코스피 2700 돌파 외국인 매수세", "content": "코스피가 외국인 대규모 순매수에 힘입어 2700선을 돌파하며 마감했다. 반도체, IT 업종이 상승을 이끌었다."},
    {"id": 2, "title": "코스피 2700선 돌파 마감", "content": "코스피 지수가 외국인 순매수 영향으로 2700선을 넘어 마감했다. 반도체와 IT 섹터가 주도했다."},
    {"id": 3, "title": "삼성전자 갤럭시 AI 기능 강화", "content": "삼성전자가 갤럭시 스마트폰에 온디바이스 AI 기능을 강화했다. 음성인식, 사진 편집 등에서 성능이 크게 향상됐다."},
]
groups = svc.group_duplicates(articles_3way)
# 1,2번은 묶이고 3번은 따로
found_dup_group = any(g["groupSize"] == 2 and set([g["representativeId"]] + g["duplicateIds"]) == {1, 2} for g in groups)
passed = len(groups) == 2 and found_dup_group
record("3기사 중 2개 중복 정확 탐지", passed, f"groups={[(g['representativeId'], g['duplicateIds']) for g in groups]}")

# ──────────────────────────────────────────────
# 케이스 8: 대표 기사 선정 (더 정보량 많은 쪽)
# ──────────────────────────────────────────────
articles_rep = [
    {"id": 1, "title": "코스피 하락", "content": "코스피가 하락했다."},  # 짧음
    {"id": 2, "title": "코스피 외국인 매도로 하락", "content": "코스피 지수가 외국인 대규모 매도세에 밀려 하락 마감했다. 기관도 동반 매도에 나서 낙폭이 확대됐다. 전문가들은 단기 조정으로 분석했다."},
    {"id": 3, "title": "코스피 외국인 매도 영향 하락 마감", "content": "코스피가 외국인 매도 영향으로 하락 마감했다. 기관 투자자들도 매도에 가세하며 지수가 내렸다. 시장 전문가들은 조정 국면으로 진단했다."},
]
groups = svc.group_duplicates(articles_rep)
if groups:
    rep_id = groups[0]["representativeId"]
    # id=2 또는 3이 대표여야 함 (더 긴/유사도 높은 쪽)
    passed = rep_id in [2, 3]
    record("대표 기사 선정 (짧은 기사 제외)", passed, f"대표 기사 id={rep_id} (2 또는 3이어야 함)")
else:
    record("대표 기사 선정", False, "그룹 없음")

# ──────────────────────────────────────────────
# 케이스 9: check_batch vs group_duplicates 일관성
# ──────────────────────────────────────────────
test_arts = [
    {"id": 1, "title": "삼성전자 HBM 공급 확대 결정", "content": "삼성전자가 엔비디아 등 AI 반도체 기업에 대한 HBM 공급을 대폭 확대하기로 결정했다. 연간 공급량을 두 배로 늘릴 계획이다."},
    {"id": 2, "title": "삼성전자 HBM 공급량 두 배 확대", "content": "삼성전자가 HBM 반도체 공급을 크게 늘리기로 했다. 엔비디아 등 AI 칩 업체에 대한 공급을 연간 두 배 수준으로 확대할 예정이다."},
    {"id": 3, "title": "카카오 AI 신사업 전략 발표", "content": "카카오가 자체 개발 AI 모델을 활용한 새로운 사업 전략을 발표했다. 검색, 커머스, 금융 서비스에 AI를 전면 통합할 계획이다."},
]

batch = svc.check_batch(test_arts)
groups = svc.group_duplicates(test_arts)

batch_dups = [r["articleId"] for r in batch if r["isDuplicate"]]
group_dups = []
for g in groups:
    group_dups.extend(g["duplicateIds"])

# 중복으로 탐지된 기사 ID가 일치하는지
passed = set(batch_dups) == set(group_dups)
record(
    "check_batch vs group_duplicates 일관성",
    passed,
    f"check_batch 중복={batch_dups} | group_duplicates 중복={group_dups}"
)

# ──────────────────────────────────────────────
# 케이스 10: 성능 측정 (100건)
# ──────────────────────────────────────────────
import random
templates = [
    ("삼성전자 반도체 실적 {n}분기 발표", "삼성전자가 {n}분기 반도체 실적을 발표했다. 영업이익은 전분기 대비 상승했으며 HBM 수요가 주요 원인이었다."),
    ("코스피 {n}포인트 상승 마감", "코스피 지수가 외국인 매수세에 {n}포인트 상승하며 마감했다. 반도체 업종이 상승을 주도했다."),
    ("현대차 {n}분기 영업이익 발표", "현대자동차가 {n}분기 영업이익을 발표했다. SUV 판매 호조와 환율 효과가 실적을 견인했다."),
]
random.seed(42)
bulk_arts = []
for i in range(100):
    tmpl = templates[i % len(templates)]
    n = random.randint(1, 9)
    bulk_arts.append({
        "id": i + 1,
        "title": tmpl[0].format(n=n),
        "content": tmpl[1].format(n=n),
    })

t0 = time.time()
groups_bulk = svc.group_duplicates(bulk_arts)
elapsed = time.time() - t0

record(
    f"100건 처리 성능 ({elapsed:.3f}s)",
    elapsed < 5.0,
    f"100건 → {len(groups_bulk)}그룹, {100 - len(groups_bulk)}건 중복 제거, costReductionRate={(100 - len(groups_bulk))/100:.2f}"
)

# ──────────────────────────────────────────────
# 최종 요약
# ──────────────────────────────────────────────
print("\n" + "="*60)
total = len(results)
passed_count = sum(1 for r in results if r["passed"])
print(f"결과: {passed_count}/{total} 통과")
print("="*60)

# JSON 저장 (노션 정리용)
with open("/home/claude/test_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "threshold_tfidf": DedupConfig.TFIDF_THRESHOLD,
        },
        "cases": results
    }, f, ensure_ascii=False, indent=2)

print("\n결과 저장 → /home/claude/test_results.json")
