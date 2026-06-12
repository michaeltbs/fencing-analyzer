"""
inference_db.py — SQLite-backed Fencer/Bout/Metrics database.

Schema:
  fencers   : Stammdaten (name, club, nation, hand, dob)
  bouts     : Gefechte (tournament, date, score, video_path)
  metrics   : Per-frame metric values (1 row per analyzed frame)
  annotations : Highlights, touchés, manual notes

Public API:
    db = FencerDB("fencing.db")
    fid = db.upsert_fencer("michael-trebis", "Michael", "Trebis", "GER", "right", "BFC Berlin")
    bid = db.create_bout(fid, opp_fid, "Doha 2026", video_path=...)
    db.insert_metrics(bid, frame_data, metrics_dict)
    db.add_annotation(bid, t=42.5, type="touche", description="Schmidt hits")
"""
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS fencers (
    id TEXT PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    first_name TEXT,
    last_name TEXT NOT NULL,
    nation TEXT,
    hand TEXT CHECK(hand IN ('left','right') OR hand IS NULL),
    club TEXT,
    birth_year INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bouts (
    id TEXT PRIMARY KEY,
    fencer_a_id TEXT NOT NULL REFERENCES fencers(id),
    fencer_b_id TEXT NOT NULL REFERENCES fencers(id),
    tournament TEXT,
    bout_date TEXT,
    venue TEXT,
    weapon TEXT DEFAULT 'epee',
    fencer_a_score INTEGER,
    fencer_b_score INTEGER,
    video_path TEXT,
    video_duration_s REAL,
    result_path TEXT,
    status TEXT DEFAULT 'pending',  -- pending, processing, complete, failed
    started_at TEXT,
    completed_at TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bouts_fencer_a ON bouts(fencer_a_id);
CREATE INDEX IF NOT EXISTS idx_bouts_fencer_b ON bouts(fencer_b_id);
CREATE INDEX IF NOT EXISTS idx_bouts_date ON bouts(bout_date);

CREATE TABLE IF NOT EXISTS metrics (
    bout_id TEXT NOT NULL REFERENCES bouts(id) ON DELETE CASCADE,
    frame_idx INTEGER NOT NULL,
    t REAL NOT NULL,  -- seconds in source video
    dist_cm REAL,
    arm_angle_m REAL, arm_angle_g REAL,
    lunge_depth_m REAL, lunge_depth_g REAL,
    body_tilt_m REAL, body_tilt_g REAL,
    vel_m REAL, vel_g REAL,
    acc_m REAL, acc_g REAL,
    hand_height_m REAL, hand_height_g REAL,
    arm_ext_m REAL, arm_ext_g REAL,
    stance_m REAL, stance_g REAL,
    head_forward_m REAL, head_forward_g REAL,
    pressure_net REAL,
    m_hip_x REAL, m_hip_y REAL, g_hip_x REAL, g_hip_y REAL,
    PRIMARY KEY (bout_id, frame_idx)
);

CREATE INDEX IF NOT EXISTS idx_metrics_bout_t ON metrics(bout_id, t);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bout_id TEXT NOT NULL REFERENCES bouts(id) ON DELETE CASCADE,
    t REAL NOT NULL,
    end_t REAL,
    type TEXT NOT NULL,  -- touche, pause, highlight, note, marker
    description TEXT,
    fencer_id TEXT REFERENCES fencers(id),
    confidence TEXT,  -- high, medium, low
    source TEXT,  -- auto, manual, subagent
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_annotations_bout_t ON annotations(bout_id, t);
CREATE INDEX IF NOT EXISTS idx_annotations_type ON annotations(bout_id, type);
"""


class FencerDB:
    """SQLite-backed fencing database."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------
    # Fencers
    # ------------------------------------------------------------------

    def upsert_fencer(self, slug, last_name, first_name=None, nation=None,
                      hand=None, club=None, birth_year=None, notes=None) -> str:
        """Insert or update fencer by slug. Returns fencer ID."""
        now = datetime.utcnow().isoformat()
        with self._conn() as c:
            row = c.execute("SELECT id FROM fencers WHERE slug = ?", (slug,)).fetchone()
            if row:
                fid = row["id"]
                c.execute("""
                    UPDATE fencers
                    SET first_name=?, last_name=?, nation=?, hand=?, club=?,
                        birth_year=?, notes=?, updated_at=?
                    WHERE id=?
                """, (first_name, last_name, nation, hand, club, birth_year,
                      notes, now, fid))
            else:
                fid = str(uuid.uuid4())
                c.execute("""
                    INSERT INTO fencers
                    (id, slug, first_name, last_name, nation, hand, club,
                     birth_year, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (fid, slug, first_name, last_name, nation, hand, club,
                      birth_year, notes, now, now))
        return fid

    def get_fencer(self, fencer_id) -> Optional[Dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM fencers WHERE id = ?", (fencer_id,)).fetchone()
            return dict(row) if row else None

    def get_fencer_by_slug(self, slug) -> Optional[Dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM fencers WHERE slug = ?", (slug,)).fetchone()
            return dict(row) if row else None

    def list_fencers(self) -> List[Dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM fencers ORDER BY last_name, first_name"
            ).fetchall()]

    # ------------------------------------------------------------------
    # Bouts
    # ------------------------------------------------------------------

    def create_bout(self, fencer_a_id, fencer_b_id, tournament=None,
                    bout_date=None, venue=None, weapon="epee",
                    fencer_a_score=None, fencer_b_score=None,
                    video_path=None, video_duration_s=None,
                    notes=None) -> str:
        """Create new bout. Returns bout ID."""
        bid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self._conn() as c:
            c.execute("""
                INSERT INTO bouts
                (id, fencer_a_id, fencer_b_id, tournament, bout_date, venue,
                 weapon, fencer_a_score, fencer_b_score, video_path,
                 video_duration_s, status, started_at, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """, (bid, fencer_a_id, fencer_b_id, tournament, bout_date, venue,
                  weapon, fencer_a_score, fencer_b_score, video_path,
                  video_duration_s, now, notes, now))
        return bid

    def get_bout(self, bout_id) -> Optional[Dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM bouts WHERE id = ?", (bout_id,)).fetchone()
            return dict(row) if row else None

    def list_bouts(self, fencer_id=None, limit=50) -> List[Dict]:
        sql = "SELECT * FROM bouts"
        params = []
        if fencer_id:
            sql += " WHERE fencer_a_id = ? OR fencer_b_id = ?"
            params = [fencer_id, fencer_id]
        sql += " ORDER BY bout_date DESC, created_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def update_bout_status(self, bout_id, status, result_path=None,
                           completed=False, fencer_a_score=None,
                           fencer_b_score=None):
        now = datetime.utcnow().isoformat()
        sets = ["status = ?"]
        vals = [status]
        if result_path:
            sets.append("result_path = ?")
            vals.append(result_path)
        if completed:
            sets.append("completed_at = ?")
            vals.append(now)
        if fencer_a_score is not None:
            sets.append("fencer_a_score = ?")
            vals.append(fencer_a_score)
        if fencer_b_score is not None:
            sets.append("fencer_b_score = ?")
            vals.append(fencer_b_score)
        vals.append(bout_id)
        with self._conn() as c:
            c.execute(f"UPDATE bouts SET {', '.join(sets)} WHERE id = ?", vals)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def insert_metrics(self, bout_id, frame_data: List[Dict],
                       m1_dist=None, m2_m_angle=None, m2_g_angle=None,
                       m3_m_lunge=None, m3_g_lunge=None,
                       m5_m_tilt=None, m5_g_tilt=None,
                       m6_m_acc=None, m6_g_acc=None,
                       m8_vel_m=None, m8_vel_g=None,
                       m9_m_hand_h=None, m9_g_hand_h=None,
                       m10_m_ext=None, m10_m_ext_g=None,
                       m11_m_stance=None, m11_g_stance=None,
                       m13_m_head=None, m13_g_head=None,
                       m16_pressure=None, m4_m_path=None, m4_g_path=None):
        """
        Insert per-frame metrics. frame_data is the source of truth for
        t/frame_idx; metric arrays are aligned with frame_data.
        """
        # Pre-extract per-frame values into a list of tuples
        def get_val(arr, i, key="cm"):
            if arr and i < len(arr):
                return arr[i].get(key)
            return None

        def get_val2(arr, i, key):
            if arr and i < len(arr):
                v = arr[i].get(key)
                return v if v is not None else None
            return None

        rows = []
        for i, f in enumerate(frame_data):
            m_hip_x = m_hip_y = g_hip_x = g_hip_y = None
            if m4_m_path and i < len(m4_m_path):
                m_hip_x = m4_m_path[i].get("x")
                m_hip_y = m4_m_path[i].get("y")
            if m4_g_path and i < len(m4_g_path):
                g_hip_x = m4_g_path[i].get("x")
                g_hip_y = m4_g_path[i].get("y")

            rows.append((
                bout_id, i, f["t"],
                get_val(m1_dist, i),
                get_val2(m2_m_angle, i, "deg"),
                get_val2(m2_g_angle, i, "deg"),
                get_val2(m3_m_lunge, i, "px"),
                get_val2(m3_g_lunge, i, "px"),
                get_val2(m5_m_tilt, i, "deg"),
                get_val2(m5_g_tilt, i, "deg"),
                m8_vel_m[i] if m8_vel_m and i < len(m8_vel_m) else None,
                m8_vel_g[i] if m8_vel_g and i < len(m8_vel_g) else None,
                get_val2(m6_m_acc, i, "acc"),
                get_val2(m6_g_acc, i, "acc"),
                get_val2(m9_m_hand_h, i, "px"),
                get_val2(m9_g_hand_h, i, "px"),
                get_val2(m10_m_ext, i, "px"),
                get_val2(m10_m_ext_g, i, "px"),
                get_val2(m11_m_stance, i, "px"),
                get_val2(m11_g_stance, i, "px"),
                get_val2(m13_m_head, i, "px"),
                get_val2(m13_g_head, i, "px"),
                get_val2(m16_pressure, i, "net_px"),
                m_hip_x, m_hip_y, g_hip_x, g_hip_y,
            ))

        with self._conn() as c:
            c.executemany("""
                INSERT OR REPLACE INTO metrics
                (bout_id, frame_idx, t, dist_cm, arm_angle_m, arm_angle_g,
                 lunge_depth_m, lunge_depth_g, body_tilt_m, body_tilt_g,
                 vel_m, vel_g, acc_m, acc_g,
                 hand_height_m, hand_height_g, arm_ext_m, arm_ext_g,
                 stance_m, stance_g, head_forward_m, head_forward_g,
                 pressure_net, m_hip_x, m_hip_y, g_hip_x, g_hip_y)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)

    def get_metrics(self, bout_id) -> List[Dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM metrics WHERE bout_id = ? ORDER BY frame_idx",
                (bout_id,)
            ).fetchall()]

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def add_annotation(self, bout_id, t, type_, description=None,
                       fencer_id=None, end_t=None, confidence=None,
                       source="auto") -> int:
        now = datetime.utcnow().isoformat()
        with self._conn() as c:
            cur = c.execute("""
                INSERT INTO annotations
                (bout_id, t, end_t, type, description, fencer_id, confidence,
                 source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (bout_id, t, end_t, type_, description, fencer_id,
                  confidence, source, now))
            return cur.lastrowid

    def get_annotations(self, bout_id, type_=None) -> List[Dict]:
        sql = "SELECT * FROM annotations WHERE bout_id = ?"
        params = [bout_id]
        if type_:
            sql += " AND type = ?"
            params.append(type_)
        sql += " ORDER BY t"
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def bulk_add_annotations(self, bout_id, annotations: List[Dict]) -> int:
        """Add many annotations at once. Each dict: t, type, description?, ..."""
        now = datetime.utcnow().isoformat()
        rows = []
        for a in annotations:
            rows.append((
                bout_id,
                a.get("t"),
                a.get("end_t"),
                a.get("type", "note"),
                a.get("description"),
                a.get("fencer_id"),
                a.get("confidence"),
                a.get("source", "auto"),
                now,
            ))
        with self._conn() as c:
            c.executemany("""
                INSERT INTO annotations
                (bout_id, t, end_t, type, description, fencer_id, confidence,
                 source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
        return len(rows)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        with self._conn() as c:
            return {
                "fencers": c.execute("SELECT COUNT(*) FROM fencers").fetchone()[0],
                "bouts": c.execute("SELECT COUNT(*) FROM bouts").fetchone()[0],
                "bouts_complete": c.execute(
                    "SELECT COUNT(*) FROM bouts WHERE status = 'complete'"
                ).fetchone()[0],
                "metrics_rows": c.execute("SELECT COUNT(*) FROM metrics").fetchone()[0],
                "annotations": c.execute("SELECT COUNT(*) FROM annotations").fetchone()[0],
            }


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "fencing.db"
    db = FencerDB(db_path)
    print(f"DB initialized at {db_path}")
    print("Schema:")
    with db._conn() as c:
        for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            print(f"  - {row['name']}")
    print(f"\nStats: {db.stats()}")
