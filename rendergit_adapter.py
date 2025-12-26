#!/usr/bin/env python3
"""
rendergit_adapter.py - Bridge between rendergit and git.symphony

A minimal adapter that converts rendergit's markdown output into
git.symphony's episodic memory format. This demonstrates how the
two projects could work together:

- rendergit: Renders git repositories as markdown narratives
- git.symphony: Explores repos using resonance, entropy, perplexity

This adapter lets Symphony "remember" rendergit explorations and
build episodic memory from them.

## Usage with rendergit

```bash
# Generate rendergit markdown
rendergit /path/to/repo > repo_story.md

# Import into Symphony's memory
python rendergit_adapter.py repo_story.md --import

# Or stream directly
rendergit /path/to/repo | python rendergit_adapter.py --stdin
```

## Integration Pattern

rendergit focuses on *narrative* - telling the story of a codebase
git.symphony focuses on *exploration* - finding patterns through metrics

Together: rendergit generates the story, Symphony remembers the journey.

Created as a gesture to @karpathy's rendergit project. 🎋
"""

import re
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter


def parse_rendergit_markdown(markdown: str) -> Dict:
    """
    Parse rendergit markdown output into structured data.

    Extracts:
    - Repository name/path
    - File hierarchy
    - Code snippets
    - Technical keywords
    - Narrative structure

    Args:
        markdown: Rendergit markdown output

    Returns:
        Dict with parsed repository data
    """
    data = {
        'repo_path': '',
        'files': [],
        'keywords': set(),
        'narrative': '',
        'commits': [],
        'structure': {}
    }

    # Extract repository path (first heading or code block)
    repo_match = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
    if repo_match:
        data['repo_path'] = repo_match.group(1).strip()

    # Extract files from markdown code blocks
    code_blocks = re.findall(r'```(\w+)?\n(.+?)\n```', markdown, re.DOTALL)
    for lang, code in code_blocks:
        data['files'].append({
            'language': lang or 'unknown',
            'content': code[:500],  # First 500 chars
            'size': len(code)
        })

    # Extract technical keywords (common programming terms)
    tech_patterns = [
        r'\b(class|function|def|async|await|import|export|const|let|var)\b',
        r'\b(python|javascript|typescript|rust|go|java|cpp)\b',
        r'\b(api|database|model|controller|service|handler|router)\b',
        r'\b(test|mock|fixture|assert|expect)\b',
        r'\b(git|commit|branch|merge|pull|push)\b'
    ]

    markdown_lower = markdown.lower()
    for pattern in tech_patterns:
        matches = re.findall(pattern, markdown_lower, re.IGNORECASE)
        data['keywords'].update(matches)

    # Extract narrative (all text outside code blocks)
    narrative_parts = re.split(r'```.*?```', markdown, flags=re.DOTALL)
    data['narrative'] = ' '.join(part.strip() for part in narrative_parts if part.strip())

    # Extract file structure (headings)
    headings = re.findall(r'^#{1,6}\s+(.+)$', markdown, re.MULTILINE)
    data['structure'] = {
        'files': [h for h in headings if '/' in h or '.' in h],
        'sections': [h for h in headings if '/' not in h and '.' not in h]
    }

    return data


def calculate_metrics(parsed_data: Dict) -> Tuple[float, float, float]:
    """
    Calculate Symphony-style metrics from rendergit data.

    Args:
        parsed_data: Parsed rendergit data

    Returns:
        Tuple of (resonance, entropy, perplexity)
    """
    import math
    from collections import Counter

    # RESONANCE: Measure of technical richness
    # More keywords + files = higher resonance
    keyword_score = min(1.0, len(parsed_data['keywords']) / 20.0)
    file_score = min(1.0, len(parsed_data['files']) / 10.0)
    resonance = (keyword_score + file_score) / 2.0

    # ENTROPY: Measure of code diversity
    # Shannon entropy of keywords
    if parsed_data['keywords']:
        keyword_counts = Counter(parsed_data['keywords'])
        total = sum(keyword_counts.values())
        entropy = 0.0
        for count in keyword_counts.values():
            p = count / total
            entropy -= p * math.log2(p)
    else:
        entropy = 0.0

    # PERPLEXITY: 2^entropy (standard formula)
    perplexity = 2 ** entropy

    return resonance, entropy, perplexity


def convert_to_episode(parsed_data: Dict, resonance: float, entropy: float,
                       perplexity: float) -> Dict:
    """
    Convert rendergit data to Symphony episode format.

    Creates an exploration episode that can be imported into
    Symphony's episodic memory database.

    Args:
        parsed_data: Parsed rendergit data
        resonance: Calculated resonance score
        entropy: Calculated entropy score
        perplexity: Calculated perplexity score

    Returns:
        Episode dict ready for Symphony import
    """
    # Extract main keyword (most common technical term)
    main_keyword = 'code'
    if parsed_data['keywords']:
        keyword_counts = Counter(parsed_data['keywords'])
        main_keyword = keyword_counts.most_common(1)[0][0]

    # Generate exploration path (key files/sections)
    path_elements = parsed_data['structure']['files'][:5]
    path_taken = ' -> '.join(path_elements) if path_elements else 'repository root'

    episode = {
        'created_at': time.time(),
        'prompt': f"Exploring {parsed_data['repo_path']}",
        'keyword': main_keyword,
        'repo_url': f"file://{parsed_data['repo_path']}",
        'path_taken': path_taken,
        'resonance': resonance,
        'entropy': entropy,
        'perplexity': perplexity,
        'user_accepted': True,  # Assume accepted since explicitly imported
        'quality': resonance,  # Use resonance as quality proxy
        'metadata': {
            'source': 'rendergit',
            'files_count': len(parsed_data['files']),
            'keywords': list(parsed_data['keywords'])[:20],  # Top 20
            'narrative_excerpt': parsed_data['narrative'][:200]
        }
    }

    return episode


def import_to_symphony(episode: Dict, db_path: str = "symphony_episodes.db"):
    """
    Import episode into Symphony's episodic memory.

    Args:
        episode: Episode dictionary
        db_path: Path to Symphony's episodes database
    """
    try:
        from episodes import EpisodicMemory, ExplorationEpisode

        memory = EpisodicMemory(Path(db_path))

        # Create ExplorationEpisode object
        exp_episode = ExplorationEpisode(
            prompt=episode['prompt'],
            keyword=episode['keyword'],
            repo_url=episode['repo_url'],
            path_taken=episode['path_taken'],
            resonance=episode['resonance'],
            entropy=episode['entropy'],
            perplexity=episode['perplexity'],
            user_accepted=episode['user_accepted'],
            timestamp=episode['created_at']
        )

        memory.observe_episode(exp_episode)
        print(f"✅ Imported episode: {episode['keyword']} (resonance: {episode['resonance']:.3f})")

    except ImportError:
        print("⚠️  Symphony episodes.py not found. Saving to JSON instead.")
        output_path = "rendergit_episodes.json"
        with open(output_path, 'w') as f:
            json.dump([episode], f, indent=2)
        print(f"💾 Saved episode to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='rendergit → git.symphony adapter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import rendergit output
  rendergit /path/to/repo > story.md
  python rendergit_adapter.py story.md --import

  # Stream from stdin
  rendergit /path/to/repo | python rendergit_adapter.py --stdin --import

  # Just convert to JSON (no import)
  python rendergit_adapter.py story.md --json

Philosophy:
  rendergit tells the story. Symphony remembers the journey. 🎋
        """
    )

    parser.add_argument(
        'input_file',
        nargs='?',
        help='Rendergit markdown file to process'
    )
    parser.add_argument(
        '--stdin',
        action='store_true',
        help='Read from stdin instead of file'
    )
    parser.add_argument(
        '--import',
        dest='do_import',
        action='store_true',
        help='Import into Symphony episodic memory'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output episode as JSON'
    )
    parser.add_argument(
        '--db',
        default='symphony_episodes.db',
        help='Path to Symphony episodes database'
    )

    args = parser.parse_args()

    # Read input
    if args.stdin:
        markdown = sys.stdin.read()
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            markdown = f.read()
    else:
        parser.print_help()
        sys.exit(1)

    # Process
    print("🎋 Parsing rendergit output...")
    parsed = parse_rendergit_markdown(markdown)

    print(f"   Repository: {parsed['repo_path']}")
    print(f"   Files: {len(parsed['files'])}")
    print(f"   Keywords: {len(parsed['keywords'])}")
    print()

    print("📊 Calculating Symphony metrics...")
    resonance, entropy, perplexity = calculate_metrics(parsed)
    print(f"   Resonance: {resonance:.3f}")
    print(f"   Entropy: {entropy:.3f}")
    print(f"   Perplexity: {perplexity:.3f}")
    print()

    # Convert to episode
    episode = convert_to_episode(parsed, resonance, entropy, perplexity)

    # Output/import
    if args.json:
        print(json.dumps(episode, indent=2))
    elif args.do_import:
        import_to_symphony(episode, args.db)
    else:
        print("💾 Episode created (use --import to add to Symphony memory)")
        print(f"   Keyword: {episode['keyword']}")
        print(f"   Path: {episode['path_taken'][:60]}...")


if __name__ == '__main__':
    main()
