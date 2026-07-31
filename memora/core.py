"""
ccdb_v5.py - Cognitive Crystal Database v5 (Production)

Scale fixes: FAISS IVF, async batch, configurable TTL, export/import, FastAPI dashboard.
"""

import sqlite3, numpy as np, faiss, time, os, pickle, json, uuid, threading
from typing import Optional, List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DOMAIN_ANCHORS = {
    "health": "medical condition diabetes symptoms treatment doctor hospital",
    "work": "software engineer programming project deadline meeting code",
    "gaming": "video game play steam xbox playstation fps rpg multiplayer",
    "family": "family mother father sister brother wife husband children",
    "casual": "daily life weather food movie music hobby chat",
}
DOMAIN_KEYWORDS = {
    "health": ["diabetes", "health", "doctor", "medicine", "walk", "exercise", "diet", "blood", "sugar", "bp", "checkup"],
    "work": ["software", "engineer", "job", "work", "project", "meeting", "deadline", "code", "programming"],
    "gaming": ["game", "play", "gaming", "xbox", "playstation", "steam", "fps"],
    "family": ["family", "mom", "dad", "mother", "father", "sister", "brother"],
}
DEDUP_THRESHOLD = 0.60
OPTIMIZE_THRESHOLD = 0.35
S_30D, S_90D, S_365D, TTL_DEFAULT = 2592000, 7776000, 31536000, 2592000


def _parse_ttl(val):
    """Parse TTL string like '7d' or 'forever' to seconds."""
    if val is None:
        return TTL_DEFAULT
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        if val == "forever":
            return float("inf")
        mul = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}
        for suffix, m in mul.items():
            if val.endswith(suffix):
                return float(val[:-len(suffix)]) * m
    return TTL_DEFAULT


class LatentAdapter:
    """Skeleton for latent-space injection v5."""
    def __init__(self, crystal_dim=384, hidden_dim=4096):
        self.projection = np.random.randn(crystal_dim, hidden_dim).astype(np.float32) * 0.01

    def project(self, e: np.ndarray) -> np.ndarray:
        return np.dot(e.astype(np.float32), self.projection)


class Memory:
    """Production CCDB: IVF index, async batch, TTL, export/import, dashboard."""

    def __init__(self, db_path: str = "memory.db", default_ttl: str = "30d"):
        self.db_path = db_path
        self.ip = db_path.replace(".db", "_faiss.bin")
        self.bp = db_path.replace(".db", "_bm25.pkl")
        self.default_ttl = _parse_ttl(default_ttl)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 384
        self.adapter = LatentAdapter(self.dim)
        self._stop_cleaner = threading.Event()
        self._lock = threading.Lock()
        self._init_db()
        self._init_index()
        self._init_bm25()
        self._init_domain_anchors()
        self._start_cleaner()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT NOT NULL,
                    text TEXT NOT NULL,
                    merged_text TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    created_at REAL NOT NULL DEFAULT 0,
                    level INTEGER NOT NULL DEFAULT 0,
                    is_session INTEGER NOT NULL DEFAULT 0,
                    session_id TEXT,
                    expires_at REAL
                )
            """)
            for col, dt in [("created_at", "REAL DEFAULT 0"), ("level", "INTEGER DEFAULT 0"),
                            ("is_session", "INTEGER DEFAULT 0"), ("session_id", "TEXT"),
                            ("expires_at", "REAL")]:
                try:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {dt}")
                except sqlite3.OperationalError:
                    pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_uu ON memories(user)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dd ON memories(domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ll ON memories(level)")

    def _init_index(self):
        if os.path.exists(self.ip):
            self.index = faiss.read_index(self.ip)
            self._trained = True
        else:
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, 100, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = 10
            self._trained = False

    def _save_index(self):
        faiss.write_index(self.index, self.ip)

    def _ensure_trained(self):
        if self._trained:
            return
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id FROM memories LIMIT 1000").fetchall()
        if len(rows) >= 100:
            ids = [r[0] for r in rows]
            embs = self._load_embeddings(ids)
            self.index.train(embs)
            self._trained = True
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self._trained = True

    def _load_embeddings(self, ids: List[int]) -> np.ndarray:
        """Load embeddings from DB for given IDs."""
        if not ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(f"SELECT merged_text FROM memories WHERE id IN ({placeholders})", ids).fetchall()
        texts = [r[0] for r in rows]
        return self._embed(texts)

    def _init_bm25(self):
        self.bm25, self.bm25_ids, self.bm25_corpus = None, [], []
        if os.path.exists(self.bp):
            with open(self.bp, "rb") as f:
                d = pickle.load(f)
                self.bm25_ids, self.bm25_corpus = d["ids"], d["corpus"]
            self.bm25 = BM25Okapi(self.bm25_corpus) if self.bm25_corpus else None

    def _save_bm25(self):
        with open(self.bp, "wb") as f:
            pickle.dump({"ids": self.bm25_ids, "corpus": self.bm25_corpus}, f)

    def _rebuild_bm25(self):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id, merged_text FROM memories").fetchall()
        self.bm25_ids = [r[0] for r in rows]
        self.bm25_corpus = [r[1].lower().split() for r in rows]
        self.bm25 = BM25Okapi(self.bm25_corpus) if self.bm25_corpus else None

    def _init_domain_anchors(self):
        self.domain_anchors = {}
        for d, t in DOMAIN_ANCHORS.items():
            e = self.model.encode([t], convert_to_numpy=True)[0].astype(np.float32)
            faiss.normalize_L2(e.reshape(1, -1))
            self.domain_anchors[d] = e

    def _start_cleaner(self):
        """Background TTL+session cleanup every 3600s."""
        def run():
            while not self._stop_cleaner.is_set():
                self._clean_expired()
                self._stop_cleaner.wait(3600)
        t = threading.Thread(target=run, daemon=True)
        t.start()

    def _clean_expired(self):
        now = time.time()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            dead = conn.execute(
                "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?", (now,)
            ).fetchall()
            if dead:
                ids = [r[0] for r in dead]
                conn.execute("DELETE FROM memories WHERE id IN " +
                           f"({','.join('?' for _ in ids)})", ids)
                try:
                    self.index.remove_ids(np.array(ids, dtype=np.int64))
                    self._save_index()
                except Exception:
                    pass

    def _detect_domain(self, emb: np.ndarray, text: str = "") -> tuple:
        best_d, best_s = "casual", 0.0
        for d, ae in self.domain_anchors.items():
            s = float(np.dot(emb, ae))
            if s > best_s:
                best_s, best_d = s, d
        if best_s >= 0.4 or not text:
            return best_d, best_s
        tl = text.lower()
        for d, kws in DOMAIN_KEYWORDS.items():
            if any(k in tl for k in kws):
                return d, 0.4
        return "casual", best_s

    def _embed(self, texts: List[str]) -> np.ndarray:
        e = self.model.encode(texts, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(e)
        return e

    def add(self, text: str, user: Optional[str] = None, session: bool = False,
            ttl: Optional[str] = None) -> int:
        """Store text. ttl='7d' for auto-delete, session=True for 1h ephemeral."""
        return self.add_many([text], user, session, ttl)[0]

    def add_many(self, texts: List[str], user: Optional[str] = None,
                 session: bool = False, ttl: Optional[str] = None) -> List[int]:
        """Batch store with semantic dedup, session, and TTL."""
        if not texts:
            return []
        user, now = user or "default", time.time()
        ttl_s = _parse_ttl(ttl) if ttl is not None else self.default_ttl
        sid = str(uuid.uuid4())[:8] if session else None
        exp = now + min(3600 if session else 9999999999, ttl_s) if ttl_s != float("inf") else None
        embs = self._embed(texts)
        ids, new_e, new_i = [], [], []

        with self._lock, sqlite3.connect(self.db_path) as conn:
            for i, text in enumerate(texts):
                dom, conf = self._detect_domain(embs[i], text)
                existing = conn.execute(
                    "SELECT id FROM memories WHERE user = ? AND text = ?", (user, text)
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE memories SET timestamp=?, domain=?, confidence=?, level=0 WHERE id=?",
                        (now, dom, float(conf), existing[0]))
                    ids.append(existing[0])
                else:
                    cur = conn.execute(
                        "INSERT INTO memories (user,text,merged_text,domain,confidence,timestamp,created_at,level,is_session,session_id,expires_at) VALUES (?,?,?,?,?,?,?,0,?,?,?)",
                        (user, text, text, dom, float(conf), now, now, int(session), sid, exp))
                    nid = cur.lastrowid
                    ids.append(nid)
                    new_e.append(embs[i])
                    new_i.append(nid)

        if new_e:
            self._ensure_trained()
            self.index.add(np.vstack(new_e))
            self._save_index()
            for nid in new_i:
                with sqlite3.connect(self.db_path) as con:
                    r = con.execute("SELECT merged_text FROM memories WHERE id=?", (nid,)).fetchone()
                if r:
                    self.bm25_ids.append(nid)
                    self.bm25_corpus.append(r[0].lower().split())
            self.bm25 = BM25Okapi(self.bm25_corpus) if self.bm25_corpus else None
            self._save_bm25()
        return ids

    def get(self, query: str, user: Optional[str] = None, domain: Optional[str] = None,
            after_date=None, before_date=None, min_confidence=None,
            top_k=5, include_bonded=True, mode="text") -> Any:
        """Hybrid search with auto-aging + optional latent projection."""
        res = self.get_many([query], user, domain, after_date, before_date,
                           min_confidence, top_k, include_bonded, mode)
        return res[0] if res else ("" if mode == "text" else [])

    def get_many(self, queries: List[str], user: Optional[str] = None,
                 domain=None, after_date=None, before_date=None,
                 min_confidence=None, top_k=5, include_bonded=True, mode="text") -> List:
        """Batch hybrid search with auto-compress, session filter, domain, bonding."""
        if self.index.ntotal == 0 or not queries:
            return [""] * len(queries) if mode == "text" else [[]] * len(queries)
        self._auto_compress()
        q_embs = self._embed(queries)
        sk = min(top_k * 4, self.index.ntotal)
        bs, idx = self.index.search(q_embs, sk)
        now = int(time.time())
        outs = []

        for qi, query in enumerate(queries):
            q_dom = self._detect_domain(q_embs[qi], query)[0]
            filt_dom = domain or q_dom
            vec_ids = [int(i) for i in idx[qi] if i != -1]
            bm_ids = []
            if self.bm25:
                bm_scores = self.bm25.get_scores(query.lower().split())
                topind = np.argsort(bm_scores)[::-1][:sk]
                bm_ids = [self.bm25_ids[i] for i in topind if bm_scores[i] > 0]
            merged_ids = []
            seen = set()
            for i in range(max(len(bm_ids), len(vec_ids))):
                if i < len(bm_ids) and bm_ids[i] not in seen:
                    merged_ids.append(bm_ids[i]); seen.add(bm_ids[i])
                if i < len(vec_ids) and vec_ids[i] not in seen:
                    merged_ids.append(vec_ids[i]); seen.add(vec_ids[i])

            params = list(merged_ids)
            where = []
            if user:
                where.append("user = ?"); params.append(user)
            where.append("(expires_at IS NULL OR expires_at > ?)"); params.append(now)
            if after_date:
                where.append("timestamp >= ?"); params.append(after_date)
            if before_date:
                where.append("timestamp <= ?"); params.append(before_date)
            if min_confidence:
                where.append("confidence >= ?"); params.append(min_confidence)
            w = " AND " + " AND ".join(where) if where else ""

            con = sqlite3.connect(self.db_path)
            ph = ",".join("?" * len(merged_ids)) if merged_ids else "NULL"
            df = " AND domain = ?" if not domain else ""
            dp = [filt_dom] if not domain else []
            rows = con.execute(
                f"SELECT id, merged_text, domain, level FROM memories WHERE id IN ({ph}){w}{df} ORDER BY timestamp DESC LIMIT ?",
                params + dp + [top_k]
            ).fetchall()
            rem = top_k - len(rows)
            if rem > 0:
                got_ids = [r[0] for r in rows]
                if got_ids:
                    exs = ",".join("?" * len(got_ids))
                    rest = con.execute(
                        f"SELECT id, merged_text, domain, level FROM memories WHERE id IN ({ph}){w} AND id NOT IN ({exs}) ORDER BY timestamp DESC LIMIT ?",
                        params + got_ids + [rem]
                    ).fetchall()
                else:
                    rest = con.execute(
                        f"SELECT id, merged_text, domain, level FROM memories WHERE id IN ({ph}){w} ORDER BY timestamp DESC LIMIT ?",
                        params + [rem]
                    ).fetchall()
            else:
                rest = []

            all_r = rows + rest
            p_ids, p_txts, p_doms = [], [], []
            st = set()
            for r in all_r:
                tag = f"{r[1]} (older)" if r[3] >= 2 else r[1]
                if tag not in st:
                    st.add(tag); p_ids.append(r[0]); p_txts.append(tag); p_doms.append(r[2])

            bonded = []
            if include_bonded and p_ids:
                ps = set(p_ids)
                for bd in set(p_doms):
                    ex = f"AND id NOT IN ({','.join('?' * len(ps))})"
                    bp = list(ps)
                    if user:
                        br = con.execute(
                            f"SELECT merged_text, level FROM memories WHERE user = ? AND domain = ? {ex} AND (expires_at IS NULL OR expires_at > ?) ORDER BY timestamp DESC LIMIT ?",
                            [user, bd] + bp + [now, top_k]
                        ).fetchall()
                    else:
                        br = con.execute(
                            f"SELECT merged_text, level FROM memories WHERE domain = ? {ex} AND (expires_at IS NULL OR expires_at > ?) ORDER BY timestamp DESC LIMIT ?",
                            [bd] + bp + [now, top_k]
                        ).fetchall()
                    for bt, bl in br:
                        tag = f"{bt} (older)" if bl >= 2 else bt
                        if tag not in st:
                            st.add(tag); bonded.append(tag)

            if mode == "latent":
                vecs = [self.adapter.project(self._embed([t])[0]) for t in p_txts + bonded]
                outs.append(vecs)
            else:
                parts = ["Primary:"]
                parts.extend(f"    {t}" for t in p_txts)
                if bonded:
                    parts.append("\nRelated:")
                    parts.extend(f"    {t}" for t in bonded[:top_k])
                outs.append("\n".join(parts))
            con.close()

        return outs

    def delete(self, user: Optional[str] = None) -> int:
        """Delete memories for user (or all). O(1)."""
        with self._lock, sqlite3.connect(self.db_path) as con:
            if user:
                rows = con.execute("SELECT id FROM memories WHERE user = ?", (user,)).fetchall()
                con.execute("DELETE FROM memories WHERE user = ?", (user,))
            else:
                rows = con.execute("SELECT id FROM memories").fetchall()
                con.execute("DELETE FROM memories")
            ids = [r[0] for r in rows]
        if ids:
            try:
                self.index.remove_ids(np.array(ids, dtype=np.int64))
                self._save_index()
            except Exception:
                pass
        self._rebuild_bm25()
        self._save_bm25()
        return len(ids)

    def info(self) -> Dict[str, Any]:
        """Return stats dict."""
        with sqlite3.connect(self.db_path) as con:
            t = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            u = con.execute("SELECT COUNT(DISTINCT user) FROM memories").fetchone()[0]
            d = con.execute("SELECT domain, COUNT(*) FROM memories GROUP BY domain").fetchall()
            l = con.execute("SELECT level, COUNT(*) FROM memories GROUP BY level").fetchall()
        return {
            "total_memories": t, "unique_users": u,
            "domains": dict(d), "levels": dict(l),
            "index_size": self.index.ntotal,
            "persisted": os.path.exists(self.ip),
            "bm25_size": len(self.bm25_corpus),
        }

    def export(self, path: str = "export.json"):
        """Export all memories to JSON with version tag."""
        data = {"version": "ccdb_v5", "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "memories": []}
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT user, text, merged_text, domain, confidence, timestamp, created_at, level, is_session, session_id, expires_at FROM memories"
            ).fetchall()
        data["memories"] = [dict(zip([
            "user", "text", "merged_text", "domain", "confidence", "timestamp",
            "created_at", "level", "is_session", "session_id", "expires_at"], r)) for r in rows]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return len(rows)

    def import_data(self, path: str, merge: bool = False):
        """Import memories from JSON. merge=True keeps existing data."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not merge:
            self.delete()
        for m in data.get("memories", []):
            self.add_many([m["text"]], user=m["user"], session=m["is_session"],
                         ttl=m.get("expires_at", None))
        return len(data.get("memories", []))

    def _auto_compress(self):
        now = time.time()
        with sqlite3.connect(self.db_path) as con:
            for lv, thr in [(0, S_30D), (1, S_90D), (2, S_365D)]:
                rows = con.execute(
                    "SELECT id, merged_text FROM memories WHERE level=? AND ?-created_at > ?",
                    (lv, now, thr)).fetchall()
                for rid, mt in rows:
                    ct = self._compress_txt(mt, lv + 1)
                    con.execute("UPDATE memories SET level=?, merged_text=? WHERE id=?",
                               (lv + 1, ct, rid))

    def _compress_txt(self, text: str, level: int) -> str:
        if level == 0:
            return text
        if level == 1:
            return text.split(".")[0].strip()
        if level == 2:
            return " ".join(text.split()[:8])[:80]
        w = text.split()[0].lower() if text.split() else "tag"
        return f"archive:{w}"

    def optimize(self) -> int:
        """Merge similar crystals from same user+domain."""
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT id, user, merged_text, domain, level FROM memories WHERE level <= 1 ORDER BY user, domain"
            ).fetchall()
        if len(rows) < 2:
            return 0
        texts = [r[2] for r in rows]
        embs = self._embed(texts)
        done = set()
        merge_ct = 0
        for i in range(len(rows)):
            if i in done:
                continue
            grp = [i]
            for j in range(i + 1, len(rows)):
                if rows[j][1] == rows[i][1] and rows[j][3] == rows[i][3] and j not in done:
                    sim = float(np.dot(embs[i], embs[j]))
                    if sim > OPTIMIZE_THRESHOLD:
                        grp.append(j); done.add(j)
            if len(grp) > 1:
                hist = "; ".join(rows[g][2][:80] for g in grp[1:])
                new_txt = f"{rows[grp[0]][2]} (history: {hist})"
                new_lvl = min(max(rows[g][4] for g in grp), 1)
                rem_ids = [rows[g][0] for g in grp if g != grp[0]]
                with sqlite3.connect(self.db_path) as con:
                    con.execute("UPDATE memories SET merged_text=?, level=?, timestamp=? WHERE id=?",
                               (new_txt, new_lvl, time.time(), rows[grp[0]][0]))
                    if rem_ids:
                        con.execute("DELETE FROM memories WHERE id IN " +
                                   f"({','.join('?' for _ in rem_ids)})", rem_ids)
                if rem_ids:
                    try:
                        self.index.remove_ids(np.array(rem_ids, dtype=np.int64))
                        self._save_index()
                    except Exception:
                        pass
                merge_ct += 1
        if merge_ct:
            self._save_index()
            self._rebuild_bm25()
            self._save_bm25()
        return merge_ct

    def dashboard(self):
        """Return FastAPI HTML dashboard."""
        import re as _re, math
        with sqlite3.connect(self.db_path) as con:
            c_t = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            c_u = con.execute("SELECT COUNT(DISTINCT user) FROM memories").fetchone()[0]
            dom_rows = con.execute("SELECT domain, COUNT(*) FROM memories GROUP BY domain").fetchall()
            recent = con.execute("SELECT id, user, merged_text, domain, strftime('%Y-%m-%d %H:%M:%S', datetime(timestamp, 'unixepoch')) FROM memories ORDER BY timestamp DESC LIMIT 15").fetchall()
        dom_json = json.dumps({d: c for d, c in dom_rows})
        recent_rows = "".join(
            f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2][:60]}</td><td>{r[3]}</td><td>{r[4]}</td></tr>"
            for r in recent
        )
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>CCDB Dashboard v5 - Cognitive Crystal DB</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@3"></script>
</head><body style="font-family:monospace;max-width:900px;margin:auto;padding:20px">
<h1>ccdb_v5 Dashboard</h1>
<p>{c_t} memories | {c_u} users | IVF index</p>
<canvas id="pieChart" width="400" height="300"></canvas>
<script>
let doms = JSON.parse('{dom_json}');
new Chart(document.getElementById('pieChart'), {{type:'pie',data:{{labels:Object.keys(doms),datasets:[{{data:Object.values(doms)}}]}}}});
</script>
<br/><table border="1" style="width:100%">
<thead><tr><th>ID</th><th>User</th><th>Memory</th><th>Domain</th><th>Time</th></tr></thead>
<tbody>{recent_rows}</tbody></table>
</body></html>"""


# ---- Main Demo ----
if __name__ == "__main__":
    mem = Memory("test.db", default_ttl="30d")

    print("=== Test 1: Dedup + Fractal Aging ===")
    mem.add("health notes")
    mem.add("Diabetes blood sugar 180 mg/dL", user="rahul")
    mem.add("Walk 3 km daily after meals", user="rahul")
    mem.add("Share price jumped 20%", user="client")
    mem.add("Favorite game: Elden Ring", user="client")
    print(mem.get("health update?", user="rahul"))

    print("\n=== Test 2: TTL ===")
    mem.add("Temp meeting notes", ttl="10s")
    print(f"Before TTL: {mem.info()['total_memories']}")
    time.sleep(12)
    mem._clean_expired()
    print(f"After 12s: {mem.info()['total_memories']}")

    print("\n=== Test 3: Scale (1000 inserts) ===")
    import random
    words = ["apple", "banana", "planet", "galaxy", "think", "code", "design", "run", "jump", "sing"]
    start = time.perf_counter()
    for _ in range(1000):
        txt = " ".join(random.choices(words, k=random.randint(3, 8)))
        mem.add(txt)
    end = time.perf_counter()
    print(f"Inserted 1000 memories in {end - start:.3f} s")
    print(f"Index size: {mem.index.ntotal}")

    start = time.perf_counter()
    g = mem.get("code design", top_k=10)
    end = time.perf_counter()
    print(f"get() latency: {(end - start)*1000:.1f}ms")
    print(f"Domain count: {mem.info()['domains']}")
    print(f"Files: test.db, test_faiss.bin, test_bm25.pkl")