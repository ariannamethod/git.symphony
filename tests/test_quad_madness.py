#!/usr/bin/env python3
"""
🔥 QUAD-MODEL MADNESS TEST 🔥
Test the ULTIMATE hybrid: LLaMA-15M + Word-NGram + Char-NGram + LSTM!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import frequency

# Load Symphony's README as the corpus
readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'README.md')
with open(readme_path, 'r', encoding='utf-8') as f:
    readme = f.read()

print("=" * 70)
print("  🔥 QUAD-MODEL MADNESS TEST - LLaMA-15M INTEGRATION! 🔥")
print("=" * 70)
print()
print(f"Training on Symphony's README ({len(readme)} chars)")
print()

# Create engine
engine = frequency.FrequencyEngine()

# Test different prompts to see which model gets used
tests = [
    ("Once upon a time", "LLaMA should handle this (story-like)"),
    ("Symphony explores", "Technical prompt"),
    ("The git repository", "Repository context"),
    ("In the beginning", "Story opening"),
]

for seed, description in tests:
    print("=" * 70)
    print(f"Test: {description}")
    print(f"Seed: '{seed}'")
    print("=" * 70)
    response = engine.generate_response(readme[:8000], seed=seed, max_length=180)
    print(f"Response: {response}")
    print()

print("=" * 70)
print("  🎊 QUAD-MODEL TEST COMPLETE! 🎊")
print("=" * 70)
print()
print("Look for the model prefix in each response:")
print("  [LLaMA-15M]   = Pure NumPy LLM (Karpathy's tinystories) - BEST! 🔥")
print("  [Word-NGram]  = Word-level n-grams (10-gram)")
print("  [LSTM]        = PyTorch tiny LSTM")
print("  [Char-NGram]  = Character-level n-grams (10-gram)")
print()
