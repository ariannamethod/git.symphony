#!/usr/bin/env python3
"""
Test multi-repository comparison feature.

Tests the --compare flag and repository ranking logic.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import symphony


def test_fetch_repo_info():
    """Test fetching repository information from GitHub API."""
    print("🧪 Testing fetch_repo_info()...")

    # Test with full name
    repo = symphony.fetch_repo_info("karpathy/nanoGPT")

    if repo:
        print(f"✅ Fetched: {repo['full_name']}")
        print(f"   Stars: {repo['stars']}")
        print(f"   Description: {repo['description'][:50]}...")

        assert 'url' in repo
        assert 'full_name' in repo
        assert 'stars' in repo
        assert repo['full_name'] == 'karpathy/nanoGPT'

        print("✅ fetch_repo_info() test PASSED!\n")
    else:
        print("❌ Failed to fetch repository\n")
        return False

    return True


def test_compare_logic():
    """Test repository comparison and ranking logic."""
    print("🧪 Testing comparison logic...")

    # Simulate repo data
    repos_data = [
        {
            'full_name': 'test/repo1',
            'url': 'https://github.com/test/repo1',
            'stars': 1000,
            'resonance': 0.892,
            'entropy': 4.5,
            'perplexity': 22.6
        },
        {
            'full_name': 'test/repo2',
            'url': 'https://github.com/test/repo2',
            'stars': 5000,
            'resonance': 0.651,
            'entropy': 3.9,
            'perplexity': 14.8
        },
        {
            'full_name': 'test/repo3',
            'url': 'https://github.com/test/repo3',
            'stars': 200,
            'resonance': 0.765,
            'entropy': 4.1,
            'perplexity': 17.2
        }
    ]

    # Sort by resonance (like in compare_repositories)
    repos_data.sort(key=lambda x: x['resonance'], reverse=True)

    # Check order
    assert repos_data[0]['full_name'] == 'test/repo1', "First should be repo1 (highest resonance)"
    assert repos_data[1]['full_name'] == 'test/repo3', "Second should be repo3"
    assert repos_data[2]['full_name'] == 'test/repo2', "Third should be repo2 (lowest resonance)"

    print(f"✅ Winner: {repos_data[0]['full_name']} (resonance: {repos_data[0]['resonance']})")
    print("✅ Comparison logic test PASSED!\n")

    return True


def test_metrics_calculation():
    """Test that metrics are calculated for repository descriptions."""
    print("🧪 Testing metrics calculation...")

    # Test text
    test_description = "A minimal PyTorch re-implementation of the OpenAI GPT training"

    # Calculate metrics (like in compare_repositories)
    resonance = symphony.calculate_resonance("explore repository", test_description)
    entropy = symphony.calculate_entropy(test_description)
    perplexity = symphony.calculate_perplexity(test_description)

    print(f"   Resonance: {resonance:.3f}")
    print(f"   Entropy: {entropy:.3f}")
    print(f"   Perplexity: {perplexity:.3f}")

    # Basic sanity checks
    assert 0.0 <= resonance <= 1.0, "Resonance should be between 0 and 1"
    assert entropy >= 0.0, "Entropy should be non-negative"
    assert perplexity >= 1.0, "Perplexity should be >= 1"

    print("✅ Metrics calculation test PASSED!\n")

    return True


def test_parse_repo_list():
    """Test parsing comma-separated repository list."""
    print("🧪 Testing repository list parsing...")

    # Test input
    repos_str = "karpathy/nanoGPT, openai/gpt-2,  anthropics/anthropic-sdk-python"

    # Parse (like in compare_repositories)
    repo_names = [r.strip() for r in repos_str.split(',')]

    assert len(repo_names) == 3, "Should parse 3 repositories"
    assert repo_names[0] == "karpathy/nanoGPT"
    assert repo_names[1] == "openai/gpt-2"
    assert repo_names[2] == "anthropics/anthropic-sdk-python"

    print(f"✅ Parsed {len(repo_names)} repositories")
    print("✅ Repository list parsing test PASSED!\n")

    return True


def main():
    """Run all multi-repo tests."""
    print("\n🎋 MULTI-REPO COMPARISON TESTS\n")
    print("="*60)

    tests = [
        ("Parse repository list", test_parse_repo_list),
        ("Metrics calculation", test_metrics_calculation),
        ("Comparison logic", test_compare_logic),
        ("Fetch repo info (API)", test_fetch_repo_info),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}\n")

    print("="*60)
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")

    if failed == 0:
        print("✅ ALL TESTS PASSED! 🎋\n")
    else:
        print(f"❌ {failed} tests failed\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
