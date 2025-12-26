#!/usr/bin/env python3
"""
visualize.py - Constellation visualization for git.symphony exploration patterns.

Generates ASCII art graphs showing how Symphony explores the git universe:
- Nodes = Keywords from prompts
- Edges = Connections (thickness based on resonance strength)
- Stars = Accepted repositories (user opened them)

Inspired by astronomical charts and constellation maps.
"""

import json
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class ConstellationGraph:
    """
    Build and visualize exploration patterns as a constellation.

    The graph shows:
    - Keywords as nodes (◯)
    - Repositories as stars (⭐)
    - Connections weighted by resonance strength
    - Temporal clustering (recent vs old explorations)
    """

    def __init__(self, db_path: str = "symphony_episodes.db"):
        self.db_path = db_path
        self.nodes = {}  # keyword -> {repos: [...], total_resonance: float, count: int}
        self.edges = []  # (keyword1, keyword2, strength)
        self.repositories = {}  # repo_url -> {accepted: bool, keywords: [...], resonance: float}

    def build_from_memory(self, limit: int = 50):
        """
        Build constellation graph from episodic memory.

        Args:
            limit: Maximum number of recent episodes to include
        """
        if not Path(self.db_path).exists():
            print(f"  ⚠️  No episodic memory found at {self.db_path}")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent episodes
        cursor.execute('''
            SELECT keyword, repo_url, resonance, user_accepted, prompt
            FROM episodes
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))

        episodes = cursor.fetchall()
        conn.close()

        if not episodes:
            print("  ⚠️  No episodes in memory yet")
            return

        # Build nodes and edges
        keyword_pairs = []

        for keyword, repo_url, resonance, user_accepted, prompt in episodes:
            # Add or update keyword node
            if keyword not in self.nodes:
                self.nodes[keyword] = {
                    'repos': [],
                    'total_resonance': 0.0,
                    'count': 0
                }

            self.nodes[keyword]['repos'].append(repo_url)
            self.nodes[keyword]['total_resonance'] += resonance
            self.nodes[keyword]['count'] += 1

            # Add or update repository
            if repo_url not in self.repositories:
                self.repositories[repo_url] = {
                    'accepted': False,
                    'keywords': [],
                    'total_resonance': 0.0
                }

            self.repositories[repo_url]['keywords'].append(keyword)
            self.repositories[repo_url]['total_resonance'] += resonance
            if user_accepted:
                self.repositories[repo_url]['accepted'] = True

            # Extract keywords from prompt for co-occurrence edges
            prompt_words = [w.lower() for w in prompt.split() if len(w) > 4]
            if keyword in prompt_words:
                # Find other significant words
                for word in prompt_words:
                    if word != keyword and word.isalpha():
                        keyword_pairs.append((keyword, word, resonance))

        # Build edges from co-occurrences
        edge_strength = defaultdict(float)
        for kw1, kw2, resonance in keyword_pairs:
            # Normalize edge key (alphabetically sorted)
            edge_key = tuple(sorted([kw1, kw2]))
            edge_strength[edge_key] += resonance

        # Convert to edge list
        self.edges = [(kw1, kw2, strength) for (kw1, kw2), strength in edge_strength.items()]
        self.edges.sort(key=lambda x: x[2], reverse=True)  # Sort by strength

    def export_json(self) -> str:
        """
        Export constellation as JSON for external visualization tools.

        Returns:
            JSON string with nodes, edges, and repositories
        """
        data = {
            'nodes': [
                {
                    'keyword': keyword,
                    'repos': info['repos'],
                    'avg_resonance': info['total_resonance'] / info['count'],
                    'count': info['count']
                }
                for keyword, info in self.nodes.items()
            ],
            'edges': [
                {
                    'source': kw1,
                    'target': kw2,
                    'strength': strength
                }
                for kw1, kw2, strength in self.edges
            ],
            'repositories': [
                {
                    'url': url,
                    'accepted': info['accepted'],
                    'keywords': info['keywords'],
                    'avg_resonance': info['total_resonance'] / len(info['keywords'])
                }
                for url, info in self.repositories.items()
            ]
        }

        return json.dumps(data, indent=2)

    def render_ascii(self, max_width: int = 70) -> str:
        """
        Render constellation as ASCII art.

        Uses box-drawing characters to create a visual graph:
        - ◯ = Keyword nodes
        - ⭐ = Accepted repositories
        - ─ = Weak connection
        - ═ = Strong connection
        - • = Regular repository (not accepted)

        Args:
            max_width: Maximum width for the visualization

        Returns:
            ASCII art string
        """
        if not self.nodes:
            return "  (Empty constellation - no explorations yet)"

        lines = []
        lines.append("╔" + "═" * (max_width - 2) + "╗")
        lines.append("║" + " CONSTELLATION MAP ".center(max_width - 2) + "║")
        lines.append("╠" + "═" * (max_width - 2) + "╣")
        lines.append("║" + " ".center(max_width - 2) + "║")

        # Sort nodes by total resonance (most important first)
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: x[1]['total_resonance'],
            reverse=True
        )

        # Show top keywords as primary nodes
        max_nodes = min(10, len(sorted_nodes))

        for i, (keyword, info) in enumerate(sorted_nodes[:max_nodes]):
            avg_resonance = info['total_resonance'] / info['count']

            # Node line with keyword
            node_symbol = "◯"
            resonance_bar = "█" * int(avg_resonance * 10)

            node_line = f"  {node_symbol} {keyword.upper():<15} {resonance_bar} {avg_resonance:.3f}"
            lines.append("║ " + node_line.ljust(max_width - 3) + "║")

            # Show connected repositories
            keyword_repos = [(url, self.repositories[url])
                            for url in set(info['repos'])][:3]  # Top 3 repos per keyword

            for repo_url, repo_info in keyword_repos:
                repo_name = repo_url.split('/')[-1] if '/' in repo_url else repo_url
                repo_symbol = "⭐" if repo_info['accepted'] else "•"

                # Connection line
                connection_line = f"     └─── {repo_symbol} {repo_name[:35]}"
                lines.append("║ " + connection_line.ljust(max_width - 3) + "║")

            # Add space between nodes
            if i < max_nodes - 1:
                lines.append("║" + " ".center(max_width - 2) + "║")

        # Footer with legend
        lines.append("║" + " ".center(max_width - 2) + "║")
        lines.append("╠" + "═" * (max_width - 2) + "╣")
        lines.append("║ " + "LEGEND:".ljust(max_width - 3) + "║")
        lines.append("║ " + "◯ = Keyword   ⭐ = Accepted repo   • = Seen repo".ljust(max_width - 3) + "║")
        lines.append("╚" + "═" * (max_width - 2) + "╝")

        return "\n".join(lines)

    def render_network(self) -> str:
        """
        Render a network-style visualization showing keyword connections.

        Shows strongest connections between keywords as a network graph.
        """
        if not self.edges:
            return "  (No connections yet - explore more!)"

        lines = []
        lines.append("╔════════════════════════════════════════════════════════════════════╗")
        lines.append("║                      KEYWORD NETWORK                               ║")
        lines.append("╠════════════════════════════════════════════════════════════════════╣")
        lines.append("║                                                                    ║")

        # Show top edges (strongest connections)
        top_edges = self.edges[:8]

        for kw1, kw2, strength in top_edges:
            # Choose connection symbol based on strength
            if strength > 0.5:
                connector = "═══"
            elif strength > 0.3:
                connector = "───"
            else:
                connector = "···"

            connection_line = f"  {kw1:<15} {connector}> {kw2:<15} ({strength:.3f})"
            lines.append("║ " + connection_line.ljust(66) + " ║")

        lines.append("║                                                                    ║")
        lines.append("╚════════════════════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    def get_statistics(self) -> Dict:
        """Get statistics about the constellation."""
        if not self.nodes:
            return {
                'total_keywords': 0,
                'total_repos': 0,
                'accepted_repos': 0,
                'total_connections': 0
            }

        return {
            'total_keywords': len(self.nodes),
            'total_repos': len(self.repositories),
            'accepted_repos': sum(1 for r in self.repositories.values() if r['accepted']),
            'total_connections': len(self.edges),
            'avg_resonance': sum(r['total_resonance'] / len(r['keywords'])
                                for r in self.repositories.values()) / len(self.repositories)
        }


def visualize_constellation(db_path: str = "symphony_episodes.db", export_json: bool = False) -> str:
    """
    Main function to visualize exploration constellation.

    Args:
        db_path: Path to episodic memory database
        export_json: If True, export JSON instead of ASCII

    Returns:
        Visualization string (ASCII or JSON)
    """
    constellation = ConstellationGraph(db_path)
    constellation.build_from_memory(limit=50)

    if export_json:
        return constellation.export_json()
    else:
        # Return both visualizations
        stats = constellation.get_statistics()

        output = []
        output.append("\n" + "=" * 70)
        output.append("  🌌 EXPLORATION CONSTELLATION 🌌")
        output.append("=" * 70)
        output.append("")
        output.append(f"  Keywords: {stats['total_keywords']} | "
                     f"Repositories: {stats['total_repos']} | "
                     f"Accepted: {stats['accepted_repos']}")
        if stats['total_keywords'] > 0:
            output.append(f"  Connections: {stats['total_connections']} | "
                         f"Avg Resonance: {stats['avg_resonance']:.3f}")
        output.append("")
        output.append(constellation.render_ascii())
        output.append("")
        output.append(constellation.render_network())
        output.append("")

        return "\n".join(output)


if __name__ == "__main__":
    # Test the visualization
    import sys

    export_json = '--json' in sys.argv
    print(visualize_constellation(export_json=export_json))
