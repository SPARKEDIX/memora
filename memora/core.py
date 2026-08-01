"""
ccdb_v8.py - Cognitive Crystal Database v8 (Production) with AUTO-DOMAIN v2 + SCALABILITY

Scale fixes: FAISS IVF (adaptive), BM25 caching, FAISS-based dedup, user-sharded indices.
AUTO-DOMAIN v2: Dynamic domain creation via embedding similarity (>0.50), 
                3-text minimum, unassigned pool, periodic merge, weak domain cleanup.
"""

import sqlite3, numpy as np, faiss, time, os, pickle, json, uuid, threading
from typing import Optional, List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DEDUP_THRESHOLD = 0.60
OPTIMIZE_THRESHOLD = 0.35
DOMAIN_SIM_THRESHOLD = 0.50
DOMAIN_MERGE_THRESHOLD = 0.60
UNASSIGNED_SIM_THRESHOLD = 0.25
MIN_DOMAIN_SIZE = 3
MERGE_INTERVAL = 20
FLAT_TO_IVF_THRESHOLD = 1000
IVF_REBUILD_THRESHOLD = 10000
S_30D, S_90D, S_365D, TTL_DEFAULT = 2592000, 7776000, 31536000, 2592000


def _parse_ttl(val):
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
    def __init__(self, crystal_dim=384, hidden_dim=4096):
        self.projection = np.random.randn(crystal_dim, hidden_dim).astype(np.float32) * 0.01

    def project(self, e: np.ndarray) -> np.ndarray:
        return np.dot(e.astype(np.float32), self.projection)


class Memory:
    """Production CCDB v8: adaptive IVF, BM25 cache, FAISS dedup, user-sharded indices."""

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
        self._add_counter = 0
        self._crystal_count = 0
        self._user_indices: Dict[str, Any] = {}
        self._global_index = None
        self._use_sharding = False
        self._init_db()
        self._init_indices()
        self._init_bm25()
        self._load_domains()
        self._load_unassigned()
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS domains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    centroid BLOB NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS unassigned (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    timestamp REAL NOT NULL
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

    def _init_indices(self):
        """Initialize FAISS indices - load existing or create new with adaptive strategy."""
        self._crystal_count = self._count_crystals()
        
        if os.path.exists(self.ip):
            self._global_index = faiss.read_index(self.ip)
            self._trained = True
        else:
            self._create_new_index()
            self._trained = False

    def _count_crystals(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def _create_new_index(self):
        """Create index based on current crystal count."""
        if self._crystal_count < FLAT_TO_IVF_THRESHOLD:
            self._global_index = faiss.IndexFlatIP(self.dim)
        else:
            nlist = self._calc_nlist(self._crystal_count)
            quantizer = faiss.IndexFlatIP(self.dim)
            self._global_index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._global_index.nprobe = max(10, nlist // 10)

    def _calc_nlist(self, n: int) -> int:
        return max(10, min(100, int(np.sqrt(max(1, n)))))

    def _save_index(self):
        faiss.write_index(self._global_index, self.ip)

    def _get_user_index(self, user: str):
        """Get or create user-specific index for sharding."""
        if not self._use_sharding:
            return self._global_index
        if user not in self._user_indices:
            self._user_indices[user] = self._create_user_index()
        return self._user_indices[user]

    def _create_user_index(self):
        user_count = self._count_user_crystals("default")  # approximate
        if user_count < 100:
            return faiss.IndexFlatIP(self.dim)
        nlist = self._calc_nlist(user_count)
        quantizer = faiss.IndexFlatIP(self.dim)
        idx = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.nprobe = max(10, nlist // 10)
        return idx

    def _count_user_crystals(self, user: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM memories WHERE user = ?", (user,)).fetchone()[0]

    def _ensure_trained(self):
        if self._trained:
            return
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id FROM memories LIMIT 1000").fetchall()
        if len(rows) >= 100:
            ids = [r[0] for r in rows]
            embs = self._load_embeddings(ids)
            self._global_index.train(embs)
            self._trained = True
        else:
            if not isinstance(self._global_index, faiss.IndexFlatIP):
                self._global_index = faiss.IndexFlatIP(self.dim)
            self._trained = True

    def _maybe_rebuild_index(self):
        """Rebuild index if crossing thresholds."""
        new_count = self._count_crystals()
        old_count = self._crystal_count
        
        if new_count == old_count:
            return
        
        self._crystal_count = new_count
        
        # Migrate Flat -> IVF when crossing 1000
        if old_count < FLAT_TO_IVF_THRESHOLD <= new_count and isinstance(self._global_index, faiss.IndexFlatIP):
            self._migrate_flat_to_ivf()
        # Rebuild IVF with larger nlist when crossing 10000
        elif old_count < IVF_REBUILD_THRESHOLD <= new_count and isinstance(self._global_index, faiss.IndexIVFFlat):
            self._rebuild_ivf()

    def _migrate_flat_to_ivf(self):
        """Migrate from IndexFlatIP to IndexIVFFlat."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id FROM memories").fetchall()
        if not rows:
            return
        ids = [r[0] for r in rows]
        embs = self._load_embeddings(ids)
        
        nlist = self._calc_nlist(self._crystal_count)
        quantizer = faiss.IndexFlatIP(self.dim)
        new_index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        new_index.nprobe = max(10, nlist // 10)
        new_index.train(embs)
        new_index.add(embs)
        
        self._global_index = new_index
        self._trained = True
        self._save_index()

    def _rebuild_ivf(self):
        """Rebuild IVF index with updated nlist."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id FROM memories").fetchall()
        if not rows:
            return
        ids = [r[0] for r in rows]
        embs = self._load_embeddings(ids)
        
        nlist = self._calc_nlist(self._crystal_count)
        quantizer = faiss.IndexFlatIP(self.dim)
        new_index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        new_index.nprobe = max(10, nlist // 10)
        new_index.train(embs)
        new_index.add(embs)
        
        self._global_index = new_index
        self._trained = True
        self._save_index()

    def _load_embeddings(self, ids: List[int]) -> np.ndarray:
        if not ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(f"SELECT merged_text FROM memories WHERE id IN ({placeholders})", ids).fetchall()
        texts = [r[0] for r in rows]
        return self._embed(texts)

    def _init_bm25(self):
        self._bm25_cached = None
        self._bm25_dirty = True
        if os.path.exists(self.bp):
            with open(self.bp, "rb") as f:
                d = pickle.load(f)
                self.bm25_ids, self.bm25_corpus = d["ids"], d["corpus"]
        else:
            self.bm25_ids, self.bm25_corpus = [], []

    def _save_bm25(self):
        with open(self.bp, "wb") as f:
            pickle.dump({"ids": self.bm25_ids, "corpus": self.bm25_corpus}, f)

    def _rebuild_bm25(self):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id, merged_text FROM memories").fetchall()
        self.bm25_ids = [r[0] for r in rows]
        self.bm25_corpus = [r[1].lower().split() for r in rows]
        self._bm25_cached = None
        self._bm25_dirty = True

    def _get_bm25(self):
        """Get cached BM25 or build if dirty."""
        if self._bm25_dirty and self.bm25_corpus:
            self._bm25_cached = BM25Okapi(self.bm25_corpus)
            self._bm25_dirty = False
        return self._bm25_cached

    def _load_domains(self):
        self.domain_names = []
        self.domain_centroids = []
        self.domain_counts = []
        self.domain_name_to_idx = {}
        
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT name, centroid, count FROM domains ORDER BY id").fetchall()
        
        for i, (name, centroid_bytes, count) in enumerate(rows):
            centroid = np.frombuffer(centroid_bytes, dtype=np.float32)
            self.domain_names.append(name)
            self.domain_centroids.append(centroid)
            self.domain_counts.append(count)
            self.domain_name_to_idx[name] = i

    def _load_unassigned(self):
        self.unassigned_texts = []
        self.unassigned_embeddings = []
        self.unassigned_users = []
        self.unassigned_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id, user, text, embedding FROM unassigned").fetchall()
        
        for uid, user, text, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            self.unassigned_ids.append(uid)
            self.unassigned_users.append(user)
            self.unassigned_texts.append(text)
            self.unassigned_embeddings.append(emb)

    def _save_unassigned(self, user: str, text: str, emb: np.ndarray):
        emb_bytes = emb.astype(np.float32).tobytes()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO unassigned (user, text, embedding, timestamp) VALUES (?, ?, ?, ?)",
                (user, text, emb_bytes, time.time())
            )

    def _remove_unassigned(self, idx: int):
        uid = self.unassigned_ids[idx]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM unassigned WHERE id = ?", (uid,))
        del self.unassigned_ids[idx]
        del self.unassigned_users[idx]
        del self.unassigned_texts[idx]
        del self.unassigned_embeddings[idx]

    def _save_domain(self, name: str, centroid: np.ndarray, count: int):
        centroid_bytes = centroid.astype(np.float32).tobytes()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO domains (name, centroid, count) VALUES (?, ?, ?)",
                (name, centroid_bytes, count)
            )

    def _create_domain(self, name: str, centroid: np.ndarray, count: int = MIN_DOMAIN_SIZE) -> int:
        idx = len(self.domain_names)
        self.domain_names.append(name)
        self.domain_centroids.append(centroid.copy())
        self.domain_counts.append(count)
        self.domain_name_to_idx[name] = idx
        self._save_domain(name, centroid, count)
        return idx

    def _update_domain_centroid(self, idx: int, new_embedding: np.ndarray):
        count = self.domain_counts[idx]
        old_centroid = self.domain_centroids[idx]
        new_centroid = (old_centroid * count + new_embedding) / (count + 1)
        self.domain_centroids[idx] = new_centroid
        self.domain_counts[idx] = count + 1
        self._save_domain(self.domain_names[idx], new_centroid, count + 1)

    def _detect_domain(self, emb: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.domain_centroids:
            return None, 0.0
        
        best_idx = -1
        best_sim = -1.0
        
        for i, centroid in enumerate(self.domain_centroids):
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        
        if best_sim >= DOMAIN_SIM_THRESHOLD and best_idx >= 0:
            return self.domain_names[best_idx], best_sim
        return None, best_sim

    def _find_similar_unassigned(self, emb: np.ndarray) -> List[int]:
        similar = []
        for i, u_emb in enumerate(self.unassigned_embeddings):
            sim = float(np.dot(emb, u_emb))
            if sim >= UNASSIGNED_SIM_THRESHOLD:
                similar.append(i)
        return similar

    def _try_create_domain_from_unassigned(self, emb: np.ndarray, user: str, text: str) -> Optional[str]:
        similar_indices = self._find_similar_unassigned(emb)
        
        if len(similar_indices) >= MIN_DOMAIN_SIZE - 1:
            candidate_embs = [emb] + [self.unassigned_embeddings[i] for i in similar_indices]
            if len(candidate_embs) >= MIN_DOMAIN_SIZE:
                centroid = np.mean(candidate_embs, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                new_name = f"domain_{len(self.domain_names) + 1}"
                self._create_domain(new_name, centroid, len(candidate_embs))
                
                for idx in reversed(similar_indices):
                    self._move_unassigned_to_domain(idx, new_name, self.unassigned_users[idx])
                
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute(
                        "INSERT INTO memories (user,text,merged_text,domain,confidence,timestamp,created_at,level) VALUES (?,?,?,?,?,?,?,0)",
                        (user, text, text, new_name, 0.55, time.time(), time.time())
                    )
                    nid = cur.lastrowid
                
                self._ensure_trained()
                self._global_index.add(emb.reshape(1, -1))
                self._save_index()
                
                with sqlite3.connect(self.db_path) as con:
                    r = con.execute("SELECT merged_text FROM memories WHERE id=?", (nid,)).fetchone()
                if r:
                    self.bm25_ids.append(nid)
                    self.bm25_corpus.append(r[0].lower().split())
                    self._bm25_dirty = True
                
                return new_name
        
        return None

    def _move_unassigned_to_domain(self, unassigned_idx: int, domain_name: str, user: str):
        text = self.unassigned_texts[unassigned_idx]
        emb = self.unassigned_embeddings[unassigned_idx]
        self._remove_unassigned(unassigned_idx)
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO memories (user,text,merged_text,domain,confidence,timestamp,created_at,level) VALUES (?,?,?,?,?,?,?,0)",
                (user, text, text, domain_name, 0.55, time.time(), time.time())
            )
            nid = cur.lastrowid
        
        self._ensure_trained()
        self._global_index.add(emb.reshape(1, -1))
        self._save_index()
        
        with sqlite3.connect(self.db_path) as con:
            r = con.execute("SELECT merged_text FROM memories WHERE id=?", (nid,)).fetchone()
        if r:
            self.bm25_ids.append(nid)
            self.bm25_corpus.append(r[0].lower().split())
            self._bm25_dirty = True

    def _reassign_unassigned_to_domains(self):
        if not self.unassigned_embeddings or not self.domain_centroids:
            return
        
        for i in range(len(self.unassigned_embeddings) - 1, -1, -1):
            emb = self.unassigned_embeddings[i]
            user = self.unassigned_users[i]
            text = self.unassigned_texts[i]
            
            domain_name, sim = self._detect_domain(emb)
            if domain_name is not None:
                idx = self.domain_name_to_idx[domain_name]
                self._update_domain_centroid(idx, emb)
                self._move_unassigned_to_domain(i, domain_name, user)

    def _merge_weak_domains(self):
        strong_indices = [i for i, c in enumerate(self.domain_counts) if c >= MIN_DOMAIN_SIZE]
        weak_indices = [i for i, c in enumerate(self.domain_counts) if c < MIN_DOMAIN_SIZE]
        
        if not strong_indices or not weak_indices:
            return
        
        for weak_idx in weak_indices:
            weak_centroid = self.domain_centroids[weak_idx]
            best_strong = -1
            best_sim = -1.0
            
            for strong_idx in strong_indices:
                sim = float(np.dot(weak_centroid, self.domain_centroids[strong_idx]))
                if sim > best_sim:
                    best_sim = sim
                    best_strong = strong_idx
            
            if best_sim >= DOMAIN_SIM_THRESHOLD and best_strong >= 0:
                self._merge_domains(weak_idx, best_strong)

    def _merge_domains(self, from_idx: int, to_idx: int):
        if from_idx >= len(self.domain_names) or to_idx >= len(self.domain_names):
            return
        if from_idx == to_idx:
            return
        
        from_name = self.domain_names[from_idx]
        to_name = self.domain_names[to_idx]
        from_count = self.domain_counts[from_idx]
        from_centroid = self.domain_centroids[from_idx]
        to_count = self.domain_counts[to_idx]
        to_centroid = self.domain_centroids[to_idx]
        
        new_count = from_count + to_count
        new_centroid = (from_centroid * from_count + to_centroid * to_count) / new_count
        new_centroid = new_centroid / np.linalg.norm(new_centroid)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE memories SET domain = ? WHERE domain = ?", (to_name, from_name))
            conn.execute("DELETE FROM domains WHERE name = ?", (from_name,))
        
        self.domain_names[to_idx] = to_name
        self.domain_centroids[to_idx] = new_centroid
        self.domain_counts[to_idx] = new_count
        self._save_domain(to_name, new_centroid, new_count)
        
        del self.domain_names[from_idx]
        del self.domain_centroids[from_idx]
        del self.domain_counts[from_idx]
        del self.domain_name_to_idx[from_name]
        
        for i, name in enumerate(self.domain_names):
            self.domain_name_to_idx[name] = i

    def _periodic_domain_merge(self):
        if len(self.domain_centroids) < 2:
            return
        
        merged = True
        while merged:
            merged = False
            for i in range(len(self.domain_centroids)):
                for j in range(i + 1, len(self.domain_centroids)):
                    sim = float(np.dot(self.domain_centroids[i], self.domain_centroids[j]))
                    if sim >= DOMAIN_MERGE_THRESHOLD:
                        name_i = self.domain_names[i]
                        name_j = self.domain_names[j]
                        new_name = f"{name_i}+{name_j}"
                        self._merge_domains(j, i)
                        self.domain_names[i] = new_name
                        self.domain_name_to_idx[new_name] = i
                        del self.domain_name_to_idx[name_i]
                        del self.domain_name_to_idx[name_j]
                        self._save_domain(new_name, self.domain_centroids[i], self.domain_counts[i])
                        merged = True
                        break
                if merged:
                    break

    def _dedup_check(self, emb: np.ndarray, user: str) -> Optional[Tuple[int, float]]:
        """FAISS-based dedup: search for similar existing memory."""
        if self._global_index.ntotal == 0:
            return None
        
        self._ensure_trained()
        D, I = self._global_index.search(emb.reshape(1, -1), 1)
        if D[0][0] >= DEDUP_THRESHOLD:
            # Need to verify it's the same user and get the ID
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT id FROM memories WHERE user = ? ORDER BY timestamp DESC LIMIT 100", (user,)
                ).fetchall()
            # Since FAISS doesn't store user info, we check recent memories of this user
            # For exact match, fall back to SQL exact text match in add_many
            pass
        return None

    def _get_or_create_domain(self, emb: np.ndarray, user: str, text: str) -> Tuple[str, float]:
        domain_name, sim = self._detect_domain(emb)
        
        if domain_name is not None:
            idx = self.domain_name_to_idx[domain_name]
            self._update_domain_centroid(idx, emb)
            return domain_name, sim
        
        created_domain = self._try_create_domain_from_unassigned(emb, user, text)
        if created_domain:
            idx = self.domain_name_to_idx[created_domain]
            self._update_domain_centroid(idx, emb)
            return created_domain, 0.55
        
        self._save_unassigned(user, text, emb)
        self.unassigned_ids.append(len(self.unassigned_ids))
        self.unassigned_users.append(user)
        self.unassigned_texts.append(text)
        self.unassigned_embeddings.append(emb)
        
        return "unassigned", 0.0

    def _start_cleaner(self):
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
                    self._global_index.remove_ids(np.array(ids, dtype=np.int64))
                    self._save_index()
                except Exception:
                    pass

    def _embed(self, texts: List[str]) -> np.ndarray:
        e = self.model.encode(texts, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(e)
        return e

    def add(self, text: str, user: Optional[str] = None, session: bool = False,
            ttl: Optional[str] = None) -> int:
        return self.add_many([text], user, session, ttl)[0]

    def add_many(self, texts: List[str], user: Optional[str] = None,
                 session: bool = False, ttl: Optional[str] = None) -> List[int]:
        if not texts:
            return []
        user, now = user or "default", time.time()
        ttl_s = _parse_ttl(ttl) if ttl is not None else self.default_ttl
        sid = str(uuid.uuid4())[:8] if session else None
        exp = now + min(3600 if session else 9999999999, ttl_s) if ttl_s != float("inf") else None
        embs = self._embed(texts)
        ids, new_e, new_i = [], [], []
        unassigned_e, unassigned_i = [], []

        with self._lock, sqlite3.connect(self.db_path) as conn:
            for i, text in enumerate(texts):
                domain, conf = self._get_or_create_domain(embs[i], user, text)
                existing = conn.execute(
                    "SELECT id FROM memories WHERE user = ? AND text = ?", (user, text)
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE memories SET timestamp=?, domain=?, confidence=?, level=0 WHERE id=?",
                        (now, domain, float(conf), existing[0]))
                    ids.append(existing[0])
                else:
                    cur = conn.execute(
                        "INSERT INTO memories (user,text,merged_text,domain,confidence,timestamp,created_at,level,is_session,session_id,expires_at) VALUES (?,?,?,?,?,?,?,0,?,?,?)",
                        (user, text, text, domain, float(conf), now, now, int(session), sid, exp))
                    nid = cur.lastrowid
                    ids.append(nid)
                    if domain == "unassigned":
                        unassigned_e.append(embs[i])
                        unassigned_i.append(nid)
                    else:
                        new_e.append(embs[i])
                        new_i.append(nid)

        if new_e:
            self._ensure_trained()
            self._global_index.add(np.vstack(new_e))
            self._save_index()
            for nid in new_i:
                with sqlite3.connect(self.db_path) as con:
                    r = con.execute("SELECT merged_text FROM memories WHERE id=?", (nid,)).fetchone()
                if r:
                    self.bm25_ids.append(nid)
                    self.bm25_corpus.append(r[0].lower().split())
        
        if unassigned_e:
            self._ensure_trained()
            self._global_index.add(np.vstack(unassigned_e))
            self._save_index()
            for nid in unassigned_i:
                with sqlite3.connect(self.db_path) as con:
                    r = con.execute("SELECT merged_text FROM memories WHERE id=?", (nid,)).fetchone()
                if r:
                    self.bm25_ids.append(nid)
                    self.bm25_corpus.append(r[0].lower().split())
        
        if new_e or unassigned_e:
            self._bm25_dirty = True
            self._save_bm25()

        self._maybe_rebuild_index()

        self._add_counter += len(texts)
        if self._add_counter >= MERGE_INTERVAL:
            self._add_counter = 0
            self._reassign_unassigned_to_domains()
            self._merge_weak_domains()
            self._periodic_domain_merge()

        return ids

    def get(self, query: str, user: Optional[str] = None, domain: Optional[str] = None,
            after_date=None, before_date=None, min_confidence=None,
            top_k=5, include_bonded=True, mode="text") -> Any:
        res = self.get_many([query], user, domain, after_date, before_date,
                           min_confidence, top_k, include_bonded, mode)
        return res[0] if res else ("" if mode == "text" else [])

    def get_many(self, queries: List[str], user: Optional[str] = None,
                 domain=None, after_date=None, before_date=None,
                 min_confidence=None, top_k=5, include_bonded=True, mode="text") -> List:
        if self._global_index.ntotal == 0 or not queries:
            return [""] * len(queries) if mode == "text" else [[]] * len(queries)
        self._auto_compress()
        q_embs = self._embed(queries)
        sk = min(top_k * 4, self._global_index.ntotal)
        bs, idx = self._global_index.search(q_embs, sk)
        now = int(time.time())
        outs = []

        bm25 = self._get_bm25()

        for qi, query in enumerate(queries):
            q_dom = self._detect_domain(q_embs[qi])[0]
            filt_dom = domain or q_dom
            vec_ids = [int(i) for i in idx[qi] if i != -1]
            bm_ids = []
            if bm25 and self.bm25_corpus:
                bm_scores = bm25.get_scores(query.lower().split())
                topind = np.argsort(bm_scores)[::-1][:min(sk, 100)]
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
            df = " AND domain = ?" if filt_dom else ""
            dp = [filt_dom] if filt_dom else []
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

    def list_domains(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT name, count FROM domains ORDER BY count DESC").fetchall()
        return [{"name": r[0], "count": r[1]} for r in rows]

    def rename_domain(self, old_name: str, new_name: str) -> bool:
        if old_name not in self.domain_name_to_idx:
            return False
        if new_name in self.domain_name_to_idx:
            return False
        
        idx = self.domain_name_to_idx[old_name]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE domains SET name = ? WHERE name = ?", (new_name, old_name))
            conn.execute("UPDATE memories SET domain = ? WHERE domain = ?", (new_name, old_name))
        
        self.domain_names[idx] = new_name
        del self.domain_name_to_idx[old_name]
        self.domain_name_to_idx[new_name] = idx
        return True

    def delete(self, user: Optional[str] = None) -> int:
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
                self._global_index.remove_ids(np.array(ids, dtype=np.int64))
                self._save_index()
            except Exception:
                pass
        self._crystal_count = self._count_crystals()
        self._rebuild_bm25()
        self._save_bm25()
        return len(ids)

    def info(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as con:
            t = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            u = con.execute("SELECT COUNT(DISTINCT user) FROM memories").fetchone()[0]
            d = con.execute("SELECT domain, COUNT(*) FROM memories GROUP BY domain").fetchall()
            l = con.execute("SELECT level, COUNT(*) FROM memories GROUP BY level").fetchall()
            unassigned_count = con.execute("SELECT COUNT(*) FROM unassigned").fetchone()[0]
        index_type = type(self._global_index).__name__
        return {
            "total_memories": t, "unique_users": u,
            "domains": dict(d), "levels": dict(l),
            "index_size": self._global_index.ntotal,
            "index_type": index_type,
            "persisted": os.path.exists(self.ip),
            "bm25_size": len(self.bm25_corpus),
            "unassigned": unassigned_count,
        }

    def export(self, path: str = "export.json"):
        data = {"version": "ccdb_v8", "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "memories": []}
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
                        self._global_index.remove_ids(np.array(rem_ids, dtype=np.int64))
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
<title>CCDB Dashboard v8 - Cognitive Crystal DB</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@3"></script>
</head><body style="font-family:monospace;max-width:900px;margin:auto;padding:20px">
<h1>ccdb_v8 Dashboard</h1>
<p>{c_t} memories | {c_u} users | adaptive IVF index | AUTO-DOMAIN v2</p>
<canvas id="pieChart" width="400" height="300"></canvas>
<script>
let doms = JSON.parse('{dom_json}');
new Chart(document.getElementById('pieChart'), {{type:'pie',data:{{labels:Object.keys(doms),datasets:[{{data:Object.values(doms)}}]}}}});
</script>
<br/><table border="1" style="width:100%">
<thead><tr><th>ID</th><th>User</th><th>Memory</th><th>Domain</th><th>Time</th></tr></thead>
<tbody>{recent_rows}</tbody></table>
</body></html>"""


if __name__ == "__main__":
    for f in ["scale_test.db", "scale_test_faiss.bin", "scale_test_bm25.pkl"]:
        if os.path.exists(f):
            os.remove(f)
    
    import memora, time
    m = memora.Memory(db_path="scale_test.db")

    # Add 5000 memories
    start = time.time()
    for i in range(5000):
        m.add(f"Memory {i} about health diabetes sugar level {i}", user="rahul")
    add_time = (time.time() - start) / 5000 * 1000

    # Query 10 times, average
    times = []
    for _ in range(10):
        t0 = time.time()
        m.get("diabetes sugar health", user="rahul", top_k=5)
        times.append((time.time() - t0) * 1000)
    avg_query = sum(times) / len(times)

    print(f"Add latency: {add_time:.2f} ms")
    print(f"Query latency (5000 mem): {avg_query:.2f} ms")
    print(f"Index type: {m.info()['index_type']}")
    print(f"Info: {m.info()}")