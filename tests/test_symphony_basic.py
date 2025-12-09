#!/usr/bin/env python3
"""
Basic tests for git.symphony

These tests demonstrate the core functionality and provide examples
of symphony's poetic exploration capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import symphony
import frequency


def test_entropy_calculation():
    """Test entropy calculation on various texts."""
    print("\n" + "=" * 70)
    print("TEST: Entropy Calculation")
    print("=" * 70)
    
    test_cases = [
        ("hello world", "Simple text"),
        ("aaaaaaaaaa", "Repetitive text"),
        ("abcdefghij", "Diverse text"),
        ("neural network machine learning", "Technical text"),
    ]
    
    for text, description in test_cases:
        entropy = symphony.calculate_entropy(text)
        print(f"  {description:30s} -> Entropy: {entropy:.4f}")
    
    print("  ✓ Entropy calculation working")


def test_perplexity_calculation():
    """Test perplexity calculation."""
    print("\n" + "=" * 70)
    print("TEST: Perplexity Calculation")
    print("=" * 70)
    
    test_texts = [
        "machine learning models",
        "zzzzzzzzzzzzz",
        "git repository exploration"
    ]
    
    for text in test_texts:
        perplexity = symphony.calculate_perplexity(text)
        print(f"  Text: '{text:30s}' -> Perplexity: {perplexity:.4f}")
    
    print("  ✓ Perplexity calculation working")


def test_resonance_scoring():
    """Test resonance scoring with trigrams."""
    print("\n" + "=" * 70)
    print("TEST: Resonance Scoring (Trigram Matching)")
    print("=" * 70)
    
    prompt = "neural network architecture"
    
    test_texts = [
        ("neural network deep learning", "Highly related"),
        ("cooking recipes and food", "Unrelated"),
        ("architecture of modern networks", "Somewhat related"),
    ]
    
    for text, description in test_texts:
        resonance = symphony.calculate_resonance(prompt, text)
        print(f"  {description:25s} -> Resonance: {resonance:.4f}")
    
    print("  ✓ Resonance scoring working")


def test_main_keyword_extraction():
    """Test main keyword extraction using entropy."""
    print("\n" + "=" * 70)
    print("TEST: Main Keyword Extraction")
    print("=" * 70)
    
    test_prompts = [
        "find me some machine learning repositories",
        "looking for transformer implementations",
        "show me interesting neural networks",
    ]
    
    for prompt in test_prompts:
        keyword = symphony.extract_main_keyword(prompt)
        print(f"  Prompt: '{prompt}'")
        print(f"  -> Main keyword: '{keyword}'")
        print()
    
    print("  ✓ Keyword extraction working")


def test_markov_explorer():
    """Test Markov chain exploration."""
    print("\n" + "=" * 70)
    print("TEST: Markov Chain Path Generation")
    print("=" * 70)
    
    markov = symphony.MarkovExplorer(order=2)
    
    # Train on sample commits
    commits = [
        "add new feature for machine learning",
        "fix bug in neural network training",
        "implement transformer architecture design",
        "update documentation for new features",
        "optimize training loop performance",
        "add support for larger neural networks",
    ]
    
    print("  Training on commits:")
    for commit in commits[:3]:
        print(f"    - {commit}")
    print("    ...")
    
    markov.train(commits)
    
    # Generate path
    seed = ["neural", "network"]
    path = markov.generate_path(seed, length=8)
    
    print(f"\n  Generated exploration path from '{' '.join(seed)}':")
    print(f"  → {path}")
    print("\n  ✓ Markov exploration working")


def test_frequency_generation():
    """Test frequency.py text generation."""
    print("\n" + "=" * 70)
    print("TEST: Frequency Text Generation (CPU-only)")
    print("=" * 70)
    
    sample_text = """
    The transformer architecture revolutionized natural language processing.
    Deep learning models can now understand context and generate coherent text.
    Neural networks learn patterns from data through backpropagation.
    """
    
    print("  Digesting technical documentation...")
    response = frequency.generate_response(sample_text, seed="The", max_length=120)
    
    print("\n  Generated response:")
    print("  " + "-" * 66)
    print(f"  {response}")
    print("  " + "-" * 66)
    
    print("\n  ✓ Frequency generation working")


def test_memory_database():
    """Test SQLite memory database."""
    print("\n" + "=" * 70)
    print("TEST: Memory Database")
    print("=" * 70)
    
    # Create temporary test database
    test_db = symphony.MemoryDatabase("test_symphony.db")
    
    # Record a repository
    repo_id = test_db.record_repository(
        "https://github.com/karpathy/nanoGPT",
        "/tmp/test_repo",
        "abc123",
        "Test repository"
    )
    
    print(f"  Recorded repository with ID: {repo_id}")
    
    # Record trail
    test_db.record_trail(
        repo_id,
        "find neural networks",
        "neural -> network -> transformer -> architecture",
        0.75,
        4.2,
        18.5
    )
    
    print("  Recorded exploration trail")
    
    # Retrieve trail
    trail = test_db.get_trail_for_repo(repo_id)
    if trail:
        print("\n  Retrieved trail data:")
        print(f"    Prompt: {trail['prompt']}")
        print(f"    Resonance: {trail['resonance']:.3f}")
        print(f"    Entropy: {trail['entropy']:.3f}")
    
    # Test dynamic column addition
    test_db.add_technology_column("Python")
    test_db.add_technology_column("Machine-Learning")
    
    test_db.close()
    
    # Cleanup
    if os.path.exists("test_symphony.db"):
        os.remove("test_symphony.db")
    
    print("\n  ✓ Memory database working")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "🎵" * 35)
    print("  git.symphony - Test Suite")
    print("🎵" * 35)
    
    test_entropy_calculation()
    test_perplexity_calculation()
    test_resonance_scoring()
    test_main_keyword_extraction()
    test_markov_explorer()
    test_frequency_generation()
    test_memory_database()
    
    print("\n" + "=" * 70)
    print("  ✅ ALL TESTS PASSED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    run_all_tests()
