#!/usr/bin/env python3
"""
git.symphony - A poetic git repository explorer that dreams through code histories.

Symphony travels through git repositories using entropy, perplexity, and resonance metrics,
building memories in SQLite and finding connections through Markov chains.
"""

import os
import re
import sys
import time
import sqlite3
import subprocess
import webbrowser
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import math
import random

# Import our frequency-based text generator
import frequency


class MemoryDatabase:
    """Manages the SQLite memory database with dynamic schema growth."""
    
    def __init__(self, db_path: str = "symphony_memory.db"):
        self.db_path = db_path
        self.max_size = 2 * 1024 * 1024  # 2MB
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize database with core tables."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Core repository memory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS repositories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                local_path TEXT,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                commit_hash TEXT,
                description TEXT,
                archived BOOLEAN DEFAULT 0
            )
        ''')
        
        # Git exploration trails - how we found things
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exploration_trails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id INTEGER,
                prompt TEXT,
                path_taken TEXT,
                resonance_score REAL,
                entropy_score REAL,
                perplexity_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (repo_id) REFERENCES repositories(id)
            )
        ''')
        
        # Commit snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS commit_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id INTEGER,
                commit_hash TEXT,
                message TEXT,
                author TEXT,
                date TEXT,
                interesting_tech TEXT,
                FOREIGN KEY (repo_id) REFERENCES repositories(id)
            )
        ''')
        
        self.conn.commit()
    
    def check_and_rotate(self):
        """Check database size and rotate if needed."""
        if not os.path.exists(self.db_path):
            return
        
        size = os.path.getsize(self.db_path)
        if size > self.max_size:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_name = f"symphony_memory_{timestamp}.db"
            
            # Archive old memories
            self.conn.close()
            os.rename(self.db_path, archive_name)
            print(f"💾 Memory rotated: {archive_name}")
            
            # Create fresh database
            self.init_database()
    
    def add_technology_column(self, tech_name: str):
        """Dynamically add column for new technology discovered."""
        safe_name = re.sub(r'[^\w]', '_', tech_name.lower())
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(f'''
                ALTER TABLE repositories 
                ADD COLUMN tech_{safe_name} INTEGER DEFAULT 0
            ''')
            self.conn.commit()
            print(f"🔬 Discovered new technology: {tech_name}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    
    def record_repository(self, url: str, local_path: str, commit_hash: str, description: str = ""):
        """Record a repository visit."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO repositories (url, local_path, commit_hash, description, last_accessed, access_count)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 
                    COALESCE((SELECT access_count + 1 FROM repositories WHERE url = ?), 1))
        ''', (url, local_path, commit_hash, description, url))
        self.conn.commit()
        return cursor.lastrowid
    
    def record_trail(self, repo_id: int, prompt: str, path: str, resonance: float, entropy: float, perplexity: float):
        """Record exploration trail."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO exploration_trails (repo_id, prompt, path_taken, resonance_score, entropy_score, perplexity_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (repo_id, prompt, path, resonance, entropy, perplexity))
        self.conn.commit()
    
    def get_trail_for_repo(self, repo_id: int) -> Optional[Dict]:
        """Get the exploration trail for a repository."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT prompt, path_taken, resonance_score, entropy_score, perplexity_score
            FROM exploration_trails
            WHERE repo_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (repo_id,))
        row = cursor.fetchone()
        if row:
            return {
                'prompt': row[0],
                'path': row[1],
                'resonance': row[2],
                'entropy': row[3],
                'perplexity': row[4]
            }
        return None
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class MarkovExplorer:
    """Markov chain-based git history explorer."""
    
    def __init__(self, order: int = 2):
        self.order = order
        self.chain = defaultdict(list)
    
    def train(self, commits: List[str]):
        """Train Markov chain on commit messages."""
        for message in commits:
            words = message.lower().split()
            for i in range(len(words) - self.order):
                state = tuple(words[i:i + self.order])
                next_word = words[i + self.order]
                self.chain[state].append(next_word)
    
    def generate_path(self, seed_words: List[str], length: int = 10) -> str:
        """Generate exploration path using Markov chain."""
        if not self.chain:
            return " -> ".join(seed_words[:3])
        
        words = seed_words[-self.order:] if len(seed_words) >= self.order else seed_words
        result = list(words)
        
        for _ in range(length):
            state = tuple(words[-self.order:])
            if state in self.chain and self.chain[state]:
                next_word = random.choice(self.chain[state])
                result.append(next_word)
                words.append(next_word)
            else:
                break
        
        return " -> ".join(result[:length])


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of text."""
    if not text:
        return 0.0
    
    counter = Counter(text.lower())
    length = len(text)
    entropy = 0.0
    
    for count in counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    
    return entropy


def calculate_perplexity(text: str) -> float:
    """Calculate perplexity of text (simplified)."""
    entropy = calculate_entropy(text)
    return 2 ** entropy


def calculate_resonance(prompt: str, text: str) -> float:
    """Calculate resonance score between prompt and text using trigrams."""
    def get_trigrams(s: str) -> set:
        s = s.lower()
        return {s[i:i+3] for i in range(len(s) - 2)}
    
    prompt_trigrams = get_trigrams(prompt)
    text_trigrams = get_trigrams(text)
    
    if not prompt_trigrams or not text_trigrams:
        return 0.0
    
    intersection = prompt_trigrams & text_trigrams
    union = prompt_trigrams | text_trigrams
    
    return len(intersection) / len(union) if union else 0.0


def extract_main_keyword(prompt: str) -> str:
    """Extract main keyword from prompt using entropy."""
    words = re.findall(r'\b\w+\b', prompt.lower())
    
    if not words:
        return ""
    
    # Calculate entropy for each word's context
    word_scores = {}
    for word in set(words):
        if len(word) > 3:  # Skip short words
            # Simple heuristic: longer words with higher entropy are more important
            word_scores[word] = len(word) * calculate_entropy(word)
    
    if not word_scores:
        return words[0] if words else ""
    
    return max(word_scores.items(), key=lambda x: x[1])[0]


def search_git_commits(repo_path: str, search_term: str, max_results: int = 20) -> List[Tuple[str, str, str]]:
    """Search git commits for matching terms using trigrams."""
    try:
        # Get commit log
        result = subprocess.run(
            ['git', '-C', repo_path, 'log', '--all', '--oneline', f'-n{max_results * 2}'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return []
        
        commits = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split(' ', 1)
            if len(parts) == 2:
                commit_hash, message = parts
                resonance = calculate_resonance(search_term, message)
                commits.append((commit_hash, message, resonance))
        
        # Sort by resonance and return top results
        commits.sort(key=lambda x: x[2], reverse=True)
        return commits[:max_results]
    
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return []


def find_readme(repo_path: str) -> Optional[str]:
    """Find the largest README file in repository."""
    readme_patterns = ['README.md', 'README.MD', 'README.txt', 'README', 'readme.md']
    best_readme = None
    best_size = 0
    
    for root, dirs, files in os.walk(repo_path):
        # Skip .git directory
        if '.git' in root:
            continue
        
        for file in files:
            if any(file.upper().startswith(pattern.upper()) for pattern in readme_patterns):
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    if size > best_size:
                        best_size = size
                        best_readme = file_path
                except OSError:
                    continue
    
    if best_readme:
        try:
            with open(best_readme, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return None
    
    return None


def draw_ascii_path(trail_data: Dict):
    """Draw ASCII art showing how symphony found the repository."""
    print("\n" + "=" * 70)
    print("  🎵 SYMPHONY'S JOURNEY 🎵")
    print("=" * 70)
    print()
    print(f"  User Prompt: '{trail_data['prompt']}'")
    print()
    print("  Metrics:")
    print(f"    → Resonance:  {trail_data['resonance']:.3f} 📡")
    print(f"    → Entropy:    {trail_data['entropy']:.3f} 🌀")
    print(f"    → Perplexity: {trail_data['perplexity']:.3f} 🧩")
    print()
    print("  Path Taken:")
    
    # Draw the path
    path_parts = trail_data['path'].split(' -> ')
    for i, part in enumerate(path_parts):
        if i == 0:
            print(f"    ╔══> {part}")
        elif i == len(path_parts) - 1:
            print(f"    ╚══> {part} ⭐")
        else:
            print(f"    ╠══> {part}")
    
    print()
    print("=" * 70 + "\n")


def confirm_open_browser() -> bool:
    """Ask user confirmation to open browser."""
    while True:
        response = input("  Open repository in browser? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("  Please answer 'y' or 'n'")


def open_repository_in_browser(url: str):
    """Open repository in web browser."""
    try:
        webbrowser.open(url)
        print(f"  🌐 Opened {url} in browser")
    except Exception as e:
        print(f"  ❌ Failed to open browser: {e}")


def explore_github(prompt: str, memory_db: MemoryDatabase) -> Optional[str]:
    """
    Main exploration function - searches GitHub for repositories.
    This is a simplified version that uses GitHub search.
    """
    # Extract main keyword
    main_keyword = extract_main_keyword(prompt)
    print(f"  🔍 Main keyword: '{main_keyword}'")
    
    # Calculate metrics
    entropy = calculate_entropy(prompt)
    perplexity = calculate_perplexity(prompt)
    
    print(f"  📊 Prompt entropy: {entropy:.3f}, perplexity: {perplexity:.3f}")
    
    # For this beta, we'll simulate finding a repository
    # In a real implementation, this would search GitHub API or local git repos
    
    # Simulated repository discovery
    simulated_repos = [
        "https://github.com/karpathy/nanoGPT",
        "https://github.com/karpathy/minGPT", 
        "https://github.com/ariannamethod/leo",
        "https://github.com/openai/gpt-2",
    ]
    
    # Pick one based on keyword
    selected_repo = simulated_repos[hash(main_keyword) % len(simulated_repos)]
    
    # Create Markov explorer for path generation
    markov = MarkovExplorer(order=2)
    
    # Simulate commit messages for training
    sample_commits = [
        "add new feature for language model",
        "fix bug in training loop",
        "implement transformer architecture",
        "update documentation and readme",
        "optimize memory usage in model",
        "add support for larger context",
    ]
    
    markov.train(sample_commits)
    
    # Generate exploration path
    prompt_words = prompt.lower().split()
    path = markov.generate_path(prompt_words, length=5)
    
    # Calculate resonance
    resonance = calculate_resonance(prompt, selected_repo)
    
    # Record in memory
    repo_id = memory_db.record_repository(
        selected_repo,
        "/tmp/symphony_repos/" + selected_repo.split('/')[-1],
        "simulated_commit_hash",
        f"Found via prompt: {prompt}"
    )
    
    memory_db.record_trail(repo_id, prompt, path, resonance, entropy, perplexity)
    
    # Get trail data for display
    trail_data = memory_db.get_trail_for_repo(repo_id)
    
    return selected_repo, trail_data


def repl_loop():
    """Main REPL loop for symphony."""
    print("=" * 70)
    print("  🎵 git.symphony - A Poetic Git Explorer 🎵")
    print("=" * 70)
    print()
    print("  Forked from Karpathy's rendergit concept")
    print("  Symphony explores git histories through dreams and resonance")
    print()
    print("  Commands:")
    print("    - Type any prompt to explore repositories")
    print("    - Type 'exit' or 'quit' to leave")
    print("=" * 70)
    print()
    
    # Initialize memory database
    memory_db = MemoryDatabase()
    
    try:
        while True:
            try:
                prompt = input("\n🎵 symphony> ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ['exit', 'quit', 'q']:
                    print("\n  👋 Farewell! Symphony dreams on...\n")
                    break
                
                print()
                print("  ♪ Symphony is exploring...")
                print()
                
                # Explore and find repository
                repo_url, trail_data = explore_github(prompt, memory_db)
                
                if repo_url and trail_data:
                    # Small pause for effect
                    time.sleep(0.5)
                    
                    # Generate response using frequency module
                    print("  💭 Generating resonance response...")
                    readme_text = f"Repository at {repo_url} - exploring {prompt}"
                    response = frequency.generate_response(readme_text, max_length=100)
                    
                    print()
                    print("  🌊 Symphony's Response:")
                    print("  " + "-" * 66)
                    print(f"  {response}")
                    print("  " + "-" * 66)
                    print()
                    
                    # Show exploration trail (default /info behavior)
                    draw_ascii_path(trail_data)
                    
                    # Confirm before opening browser
                    if confirm_open_browser():
                        open_repository_in_browser(repo_url)
                    else:
                        print("  📝 Repository recorded in memory.")
                else:
                    print("  ❌ No resonant repositories found for this prompt.")
                
                # Check and rotate database if needed
                memory_db.check_and_rotate()
                
            except KeyboardInterrupt:
                print("\n  👋 Farewell! Symphony dreams on...\n")
                break
            except Exception as e:
                print(f"\n  ⚠️  Error: {e}")
                continue
    
    finally:
        memory_db.close()


def main():
    """Entry point for symphony."""
    repl_loop()


if __name__ == "__main__":
    main()
