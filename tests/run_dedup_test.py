"""
로컬 PC에서 실행하는 실DB 중복 판별 테스트
사용법: python tests/run_dedup_test.py [options]

Options:
  --limit 500            기사 수
  --threshold 0.90       유사도 임계값
  --experiment all       default|title_weight|prefix_clean|all
  --threshold-sweep      0.85~0.95 sweep
  --date-filter          날짜 ±1일 필터 적용
  --date-window 1        날짜 필터 범위 (일)
  --category 정치        카테고리 필터

환경변수 설정 (.env 또는 직접 export):
  SSH_HOST, SSH_PORT, SSH_USER, SSH_PASS
  DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME

의존성: pip install pymysql sshtunnel==0.4.0 paramiko==3.4.0 scikit-learn numpy python-dotenv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
import time
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── SSH / DB 설정 (환경변수로 주입) ──────────────────────────────
SSH_HOST = os.getenv("SSH_HOST", "")
SSH_PORT = int(os.getenv("SSH_PORT", "22"))
SSH_USER = os.getenv("SSH_USER", "")
SSH_PASS = os.getenv("SSH_PASS", "")

DB_HOST  = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT  = int(os.getenv("DB_PORT", "3306"))
DB_USER  = os.getenv("DB_USER", "")
DB_PASS  = os.getenv("DB_PASS", "")
DB_NAME  = os.getenv("DB_NAME", "shortify")
# ───────────────────────────────────────────────────────────────


def connect_db():
    from sshtunnel import SSHTunnelForwarder
    import pymysql

    if not SSH_HOST or not SSH_USER:
        raise ValueError("SSH_HOST, SSH_USER 환경변수를 설정해주세요. (.env 파일 또는 export)")

    tunnel = SSHTunnelForwarder(
        (SSH_HOST, SSH_PORT),
        ssh_username=SSH_USER,
        ssh_password=SSH_PASS,
        remote_bind_address=(DB_HOST, DB_PORT),
    )
    tunnel.start()
    conn = pymysql.connect(
        host="127.0.0.1",
        port=tunnel.local_bind_port,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    return tunnel, conn


def load_articles(conn, limit=500, category=None):
    where = "WHERE am.status IN ('SUMMARY_DONE', 'SUMMARY_FAILED')"
    if category:
        where += f" AND am.category = '{category}'"
    sql = f"""
        SELECT oa.id, am.title, oa.content, am.press, am.category, am.published_at
        FROM original_article oa
        JOIN article_meta am ON oa.url = am.url
        {where}
        ORDER BY am.published_at DESC
        LIMIT {limit}
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [{
        "id": r["id"],
        "title": r["title"] or "",
        "content": r["content"] or "",
        "press": r["press"] or "",
        "category": r["category"] or "",
        "published_at": str(r["published_at"]),
    } for r in rows]


BOILERPLATE_RE = re.compile(
    r"(※.{0,100}|▶.{0,100}|▷.{0,100}|"
    r"이메일\s*:\s*\S+|카카오톡\s*:\s*\S+|"
    r"[가-힣a-zA-Z]+@[a-zA-Z0-9.]+\.[a-zA-Z]{2,}|"
    r"https?://\S+|무단.{0,20}금지|저작권.{0,30}|"
    r"\[\s*사진\s*=.{0,50}\]|\(\s*[가-힣]+\s*제공\s*\))", re.MULTILINE
)
PREFIX_RE = re.compile(r"^\s*\[(속보|단독|긴급|종합|업데이트)\]\s*")

def preprocess_default(title, content, max_chars=500):
    return title + " " + content[:max_chars]

def preprocess_title_weight(title, content, max_chars=500):
    clean = BOILERPLATE_RE.sub(" ", content[:max_chars * 2])[:max_chars]
    title_norm = PREFIX_RE.sub("", title).strip()
    return (title_norm + " ") * 3 + clean

def preprocess_prefix_clean(title, content, max_chars=500):
    title_norm = PREFIX_RE.sub("", title).strip()
    return title_norm + " " + content[:max_chars]

EXPERIMENTS = {
    "default":      preprocess_default,
    "title_weight": preprocess_title_weight,
    "prefix_clean": preprocess_prefix_clean,
}


def parse_date(s):
    if not s or s == "None":
        return None
    try:
        return datetime.fromisoformat(str(s).split(".")[0])
    except:
        return None


def run_tfidf(articles, preprocess_fn, threshold, date_window_days=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts  = [preprocess_fn(a["title"], a["content"]) for a in articles]
    ids    = [a["id"] for a in articles]
    dates  = [parse_date(a.get("published_at")) for a in articles]
    n      = len(texts)
    window = timedelta(days=date_window_days) if date_window_days else None

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), min_df=1, max_features=10000)
    m   = vec.fit_transform(texts)
    sim = cosine_similarity(m)

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y): parent[find(x)] = find(y)

    skipped = 0
    for i in range(n):
        for j in range(i + 1, n):
            if window and dates[i] and dates[j]:
                if abs(dates[i] - dates[j]) > window:
                    skipped += 1
                    continue
            if sim[i][j] >= threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    result = []
    for root, members in groups.items():
        if len(members) == 1:
            rep_idx, max_sim = members[0], 0.0
        else:
            avg_sims = [np.mean([sim[i][j] for j in members if j != i]) for i in members]
            rep_idx  = members[int(np.argmax(avg_sims))]
            max_sim  = float(np.max(avg_sims))
        result.append({
            "representativeId": ids[rep_idx],
            "duplicateIds":     [ids[m] for m in members if m != rep_idx],
            "groupSize":        len(members),
            "maxSimilarity":    round(max_sim, 4),
        })
    return result, sim, skipped


def analyze_groups(groups, articles, skipped=0):
    art_map    = {a["id"]: a for a in articles}
    total      = len(articles)
    unique     = len(groups)
    dup_groups = [g for g in groups if g["groupSize"] > 1]
    max_size   = max(g["groupSize"] for g in groups)

    print(f"  총 {total}건 → {unique}그룹")
    print(f"  중복그룹: {len(dup_groups)}개 | 중복제거: {total-unique}건 ({(total-unique)/total*100:.1f}%)")
    print(f"  최대 그룹 크기: {max_size}" + ("  ⚠️  과탐지 의심" if max_size > 10 else " ✅"))
    if skipped:
        print(f"  날짜 필터로 스킵: {skipped:,}쌍")

    print(f"\n  중복 그룹 상세 (size > 1):")
    for i, g in enumerate(sorted(dup_groups, key=lambda x: -x["groupSize"])[:15]):
        rep = art_map.get(g["representativeId"], {})
        print(f"    [{i+1}] size={g['groupSize']} sim={g['maxSimilarity']}")
        print(f"      ✅ [{rep.get('press','')}] {rep.get('title','')[:60]}")
        for did in g["duplicateIds"][:3]:
            d = art_map.get(did, {})
            print(f"      ❌ [{d.get('press','')}] {d.get('title','')[:60]}")
        if len(g["duplicateIds"]) > 3:
            print(f"      ... 외 {len(g['duplicateIds'])-3}건")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",           type=int,   default=500)
    parser.add_argument("--threshold",       type=float, default=0.90)
    parser.add_argument("--experiment",      type=str,   default="title_weight")
    parser.add_argument("--category",        type=str,   default=None)
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--date-filter",     action="store_true")
    parser.add_argument("--date-window",     type=int,   default=1)
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] DB 연결 중...")
    tunnel, conn = connect_db()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 기사 로드 중... (limit={args.limit})")
    articles = load_articles(conn, limit=args.limit, category=args.category)
    conn.close(); tunnel.stop()
    print(f"  → {len(articles)}건 로드 완료\n")

    date_window = args.date_window if args.date_filter else None

    if args.threshold_sweep:
        print(f"=== Threshold Sweep (title_weight, date_filter={'ON ±'+str(date_window)+'일' if date_window else 'OFF'}) ===")
        fn = EXPERIMENTS["title_weight"]
        for th in [0.85, 0.87, 0.88, 0.90, 0.92, 0.95]:
            t0 = time.time()
            groups, _, skipped = run_tfidf(articles, fn, th, date_window)
            elapsed    = time.time() - t0
            dup_groups = [g for g in groups if g["groupSize"] > 1]
            max_size   = max(g["groupSize"] for g in groups)
            total      = len(articles)
            print(f"  th={th}: {len(groups)}그룹 | 중복그룹={len(dup_groups)} | "
                  f"제거={total-len(groups)}건({(total-len(groups))/total*100:.0f}%) | "
                  f"max_size={max_size} | {elapsed:.2f}s"
                  + (f" | 스킵={skipped:,}쌍" if skipped else ""))
        print()

    if args.date_filter:
        print(f"=== 날짜 필터 ON vs OFF 비교 (title_weight, threshold={args.threshold}) ===")
        fn = EXPERIMENTS["title_weight"]
        g_off, _, _     = run_tfidf(articles, fn, args.threshold, None)
        g_on,  _, sk_on = run_tfidf(articles, fn, args.threshold, date_window)
        total = len(articles)
        print(f"  날짜필터 OFF: {len(g_off)}그룹 | 중복제거={total-len(g_off)}건 | max_size={max(g['groupSize'] for g in g_off)}")
        print(f"  날짜필터 ON (±{date_window}일): {len(g_on)}그룹 | 중복제거={total-len(g_on)}건 | max_size={max(g['groupSize'] for g in g_on)} | 스킵={sk_on:,}쌍")
        print()
        print(f"=== 날짜필터 ON (±{date_window}일) 상세 ===")
        analyze_groups(g_on, articles, sk_on)
        return

    exps = EXPERIMENTS if args.experiment == "all" else {args.experiment: EXPERIMENTS[args.experiment]}
    for name, fn in exps.items():
        label = name + (f" + date±{date_window}일" if date_window else "")
        print(f"=== 실험: {label} | threshold={args.threshold} ===")
        t0 = time.time()
        groups, _, skipped = run_tfidf(articles, fn, args.threshold, date_window)
        print(f"  처리 시간: {time.time()-t0:.3f}s")
        analyze_groups(groups, articles, skipped)
        print()


if __name__ == "__main__":
    main()
