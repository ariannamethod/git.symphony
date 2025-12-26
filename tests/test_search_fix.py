#!/usr/bin/env python3
"""
Quick test script to verify GitHub search is working and returning diverse results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import symphony
from episodes import EpisodicMemory

def test_search(keyword):
    """Test search for a keyword and print results."""
    print(f"\n{'='*70}")
    print(f"Testing search for: '{keyword}'")
    print('='*70)

    # Extract keyword
    main_keyword = symphony.extract_main_keyword(keyword)
    print(f"Main keyword extracted: '{main_keyword}'")

    # Search GitHub
    repos = symphony.search_github_repos(main_keyword, max_results=5)

    if repos:
        print(f"\nFound {len(repos)} repositories:")
        for i, repo in enumerate(repos, 1):
            print(f"  {i}. {repo['full_name']} ({repo['stars']}⭐)")
            if repo['description']:
                print(f"     {repo['description'][:70]}...")

        # Calculate resonance scores
        print(f"\nResonance scores with prompt '{keyword}':")
        for repo in repos[:3]:
            repo_text = f"{repo['name']} {repo['description']}"
            resonance = symphony.calculate_resonance(keyword, repo_text)
            print(f"  {repo['full_name']}: {resonance:.3f}")
    else:
        print("No repositories found")

def main():
    print("=" * 70)
    print("  Testing git.symphony GitHub Search Fix")
    print("=" * 70)
    print("\nThis test verifies that different keywords return different repos")
    print("(not just Karpathy's repos every time!)\n")

    # Test different search terms
    test_search("rust programming language")
    test_search("machine learning transformers")
    test_search("web framework python")
    test_search("database sqlite")
    test_search("neural network pytorch")

    print("\n" + "=" * 70)
    print("  Test Complete!")
    print("=" * 70)
    print("\n✅ If you see diverse repositories above (not all Karpathy),")
    print("   then the GitHub API search is working correctly!\n")

if __name__ == "__main__":
    main()
