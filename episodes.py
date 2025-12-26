#!/usr/bin/env python3
"""
episodes.py - Episodic memory for Symphony's explorations

Symphony remembers each journey: prompt + path + repository + metrics.
This is her episodic memory - structured recall of exploration experiences.

Inspired by Leo's episodic RAG architecture but adapted for git exploration.
No external APIs. Just local SQLite + simple similarity matching.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class ExplorationEpisode:
    """One exploration journey in Symphony's memory."""
    prompt: str
    keyword: str
    repo_url: str
    path_taken: str
    resonance: float
    entropy: float
    perplexity: float
    user_accepted: bool  # Did user open the repository?
    timestamp: float


def cosine_distance(a: List[float], b: List[float]) -> float:
    """Compute cosine distance between two vectors (1 - cosine similarity)."""
    if len(a) != len(b):
        return 1.0
        
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    
    if na == 0 or nb == 0:
        return 1.0
        
    similarity = dot / (na * nb)
    return 1.0 - similarity


def text_similarity(text1: str, text2: str) -> float:
    """Simple text similarity using character trigrams."""
    def get_trigrams(s: str) -> set:
        s = s.lower()
        return {s[i:i+3] for i in range(len(s) - 2)}
    
    t1 = get_trigrams(text1)
    t2 = get_trigrams(text2)
    
    if not t1 or not t2:
        return 0.0
    
    intersection = t1 & t2
    union = t1 | t2
    
    return len(intersection) / len(union) if union else 0.0


class EpisodicMemory:
    """
    Episodic memory for Symphony's exploration journeys.
    
    Stores (prompt, path, repo, metrics, success) as episodes in SQLite.
    Provides similarity search to find past successful explorations.
    """
    
    def __init__(self, db_path: Path | None = None) -> None:
        """
        Args:
            db_path: Path to SQLite DB (default: symphony_episodes.db)
        """
        if db_path is None:
            db_path = Path("symphony_episodes.db")
            
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self._ensure_schema()
        except Exception as e:
            print(f"⚠️  Failed to initialize episodic memory: {e}")
            
    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      REAL NOT NULL,
                prompt          TEXT NOT NULL,
                keyword         TEXT NOT NULL,
                repo_url        TEXT NOT NULL,
                path_taken      TEXT NOT NULL,

                -- exploration metrics
                resonance       REAL NOT NULL,
                entropy         REAL NOT NULL,
                perplexity      REAL NOT NULL,

                -- success indicator
                user_accepted   INTEGER NOT NULL,

                -- computed quality score
                quality         REAL NOT NULL,

                -- memory decay system
                memory_strength REAL NOT NULL DEFAULT 1.0,
                last_accessed   REAL NOT NULL
            )
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_created
            ON episodes(created_at)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_keyword
            ON episodes(keyword)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_quality
            ON episodes(quality DESC)
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_strength
            ON episodes(memory_strength DESC)
        """)

        # Migrate existing rows to have memory_strength and last_accessed
        try:
            cur.execute("""
                UPDATE episodes
                SET memory_strength = 1.0, last_accessed = created_at
                WHERE memory_strength IS NULL
            """)
        except sqlite3.OperationalError:
            pass  # Columns don't exist yet in old databases

        conn.commit()
        conn.close()
        
    def observe_episode(self, episode: ExplorationEpisode) -> None:
        """
        Record one exploration episode.
        
        Quality is computed as: resonance * (1 if accepted else 0.3)
        This weighs successful explorations higher.
        """
        try:
            # Compute quality score
            quality = episode.resonance * (1.0 if episode.user_accepted else 0.3)
            
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO episodes (
                    created_at, prompt, keyword, repo_url, path_taken,
                    resonance, entropy, perplexity,
                    user_accepted, quality, memory_strength, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.timestamp,
                episode.prompt,
                episode.keyword.lower(),
                episode.repo_url,
                episode.path_taken,
                episode.resonance,
                episode.entropy,
                episode.perplexity,
                1 if episode.user_accepted else 0,
                quality,
                1.0,  # Initial memory strength
                episode.timestamp,  # last_accessed = created_at initially
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Failed to record episode: {e}")
            
    def query_similar_prompts(
        self,
        prompt: str,
        keyword: str,
        top_k: int = 5,
        min_quality: float = 0.3,
        min_strength: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Find past episodes with similar prompts and keywords.

        Returns list of episodes sorted by similarity and quality.
        Only returns memories with sufficient strength (not too faded).
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Get episodes with similar keywords or high quality
            # Filter by memory strength to exclude faded memories
            cur.execute("""
                SELECT * FROM episodes
                WHERE quality >= ? AND memory_strength >= ? AND (
                    keyword = ? OR
                    user_accepted = 1
                )
                ORDER BY quality DESC, memory_strength DESC, created_at DESC
                LIMIT 100
            """, (min_quality, min_strength, keyword.lower()))
            
            rows = cur.fetchall()
            conn.close()
            
            if not rows:
                return []
                
            # Score by text similarity
            scored: List[Tuple[float, Dict[str, Any]]] = []
            
            for row in rows:
                # Combine prompt similarity and keyword match
                prompt_sim = text_similarity(prompt, row["prompt"])
                keyword_match = 1.0 if row["keyword"] == keyword.lower() else 0.0
                
                # Combined score: weighted average
                combined_score = (
                    0.5 * prompt_sim + 
                    0.3 * keyword_match +
                    0.2 * row["quality"]
                )
                
                scored.append((combined_score, {
                    "episode_id": row["id"],
                    "created_at": row["created_at"],
                    "prompt": row["prompt"],
                    "keyword": row["keyword"],
                    "repo_url": row["repo_url"],
                    "path_taken": row["path_taken"],
                    "resonance": row["resonance"],
                    "entropy": row["entropy"],
                    "perplexity": row["perplexity"],
                    "user_accepted": bool(row["user_accepted"]),
                    "quality": row["quality"],
                    "similarity_score": combined_score,
                }))
                
            # Sort by combined score (highest = most similar)
            scored.sort(key=lambda x: x[0], reverse=True)
            
            return [item[1] for item in scored[:top_k]]
            
        except Exception as e:
            print(f"⚠️  Failed to query episodes: {e}")
            return []
            
    def query_by_metrics(
        self,
        resonance: float,
        entropy: float,
        perplexity: float,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find episodes with similar metric signatures.
        
        Useful for finding repositories that match the "feel" of the current search.
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            cur.execute("""
                SELECT * FROM episodes
                WHERE user_accepted = 1
                ORDER BY quality DESC
                LIMIT 100
            """)
            
            rows = cur.fetchall()
            conn.close()
            
            if not rows:
                return []
                
            # Score by metric distance
            query_vec = [resonance, entropy, perplexity / 20.0]  # Normalize perplexity
            scored: List[Tuple[float, Dict[str, Any]]] = []
            
            for row in rows:
                episode_vec = [
                    row["resonance"], 
                    row["entropy"], 
                    row["perplexity"] / 20.0
                ]
                
                distance = cosine_distance(query_vec, episode_vec)
                
                scored.append((distance, {
                    "episode_id": row["id"],
                    "repo_url": row["repo_url"],
                    "resonance": row["resonance"],
                    "entropy": row["entropy"],
                    "perplexity": row["perplexity"],
                    "quality": row["quality"],
                    "distance": distance,
                }))
                
            # Sort by distance (lowest = most similar)
            scored.sort(key=lambda x: x[0])
            
            return [item[1] for item in scored[:top_k]]
            
        except Exception as e:
            print(f"⚠️  Failed to query by metrics: {e}")
            return []
            
    def get_best_cache_hit(self, keyword: str, min_strength: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Quick cache lookup for a keyword.

        Returns the highest quality episode for this keyword that was accepted.
        Only returns memories with sufficient strength (not too faded).
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute("""
                SELECT * FROM episodes
                WHERE keyword = ? AND user_accepted = 1 AND memory_strength >= ?
                ORDER BY quality DESC, memory_strength DESC, created_at DESC
                LIMIT 1
            """, (keyword.lower(), min_strength))
            
            row = cur.fetchone()
            conn.close()
            
            if row:
                return {
                    "repo_url": row["repo_url"],
                    "path_taken": row["path_taken"],
                    "resonance": row["resonance"],
                    "quality": row["quality"],
                    "created_at": row["created_at"],
                }
            
            return None
            
        except Exception:
            return None
            
    def decay_memories(self, decay_rate: float = 0.001, min_strength: float = 0.1) -> int:
        """
        Apply time-based decay to memory strength.

        Memories naturally fade over time unless refreshed.
        Decay formula: strength *= exp(-decay_rate * days_since_access)

        Args:
            decay_rate: Rate of decay per day (default: 0.001 = slow fade)
            min_strength: Minimum strength floor (memories never go below this)

        Returns:
            Number of memories decayed
        """
        try:
            current_time = time.time()
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            # Apply exponential decay based on time since last access
            cur.execute("""
                UPDATE episodes
                SET memory_strength = MAX(?, memory_strength * EXP(-? * (? - last_accessed) / 86400.0))
                WHERE memory_strength > ?
            """, (min_strength, decay_rate, current_time, min_strength))

            affected_rows = cur.rowcount
            conn.commit()
            conn.close()

            return affected_rows

        except Exception as e:
            print(f"⚠️  Failed to decay memories: {e}")
            return 0

    def refresh_memory(self, episode_id: int, boost: float = 0.2, max_strength: float = 1.5) -> None:
        """
        Refresh a memory when accessed, increasing its strength.

        This implements the "use it or lose it" pattern - frequently accessed
        memories become stronger and resist decay better.

        Args:
            episode_id: ID of the episode to refresh
            boost: How much to increase strength (default: 0.2)
            max_strength: Maximum strength cap (default: 1.5)
        """
        try:
            current_time = time.time()
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            cur.execute("""
                UPDATE episodes
                SET
                    memory_strength = MIN(?, memory_strength + ?),
                    last_accessed = ?
                WHERE id = ?
            """, (max_strength, boost, current_time, episode_id))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"⚠️  Failed to refresh memory: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about stored episodes."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            
            cur.execute("""
                SELECT 
                    COUNT(*) as total_episodes,
                    SUM(user_accepted) as accepted_episodes,
                    AVG(quality) as avg_quality,
                    AVG(resonance) as avg_resonance,
                    COUNT(DISTINCT keyword) as unique_keywords,
                    COUNT(DISTINCT repo_url) as unique_repos
                FROM episodes
            """)
            
            row = cur.fetchone()
            conn.close()
            
            if row:
                return {
                    "total_episodes": row[0],
                    "accepted_episodes": row[1] or 0,
                    "avg_quality": row[2] or 0.0,
                    "avg_resonance": row[3] or 0.0,
                    "unique_keywords": row[4] or 0,
                    "unique_repos": row[5] or 0,
                }
            
            return {}
            
        except Exception:
            return {}


__all__ = ["EpisodicMemory", "ExplorationEpisode"]
