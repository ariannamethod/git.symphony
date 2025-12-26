#!/usr/bin/env python3
"""
Test the UPGRADED frequency.py madness!
Three models in one: Word n-grams, Char n-grams, and LSTM!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import frequency

# Load Symphony's own README as the corpus
readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'README.md')
with open(readme_path, 'r', encoding='utf-8') as f:
    symphony_readme = f.read()

print("=" * 70)
print("  🎄 CHRISTMAS MADNESS TEST - FREQUENCY.PY UPGRADED 🎄")
print("=" * 70)
print()
print(f"Training on Symphony's README ({len(symphony_readme)} chars)")
print()

# Create engine
engine = frequency.FrequencyEngine()

# Test 1: Generate response about git exploration
print("=" * 70)
print("Test 1: Generate response about 'git exploration'")
print("=" * 70)
response1 = engine.generate_response(
    symphony_readme[:5000],
    seed="Symphony explores",
    max_length=200
)
print(response1)
print()

# Test 2: Generate response about neural networks
print("=" * 70)
print("Test 2: Generate response about 'transformers and attention'")
print("=" * 70)
response2 = engine.generate_response(
    symphony_readme[2000:7000],
    seed="The transformer",
    max_length=200
)
print(response2)
print()

# Test 3: Generate response about memory
print("=" * 70)
print("Test 3: Generate response about 'episodic memory'")
print("=" * 70)
response3 = engine.generate_response(
    symphony_readme[5000:10000],
    seed="Memory systems",
    max_length=200
)
print(response3)
print()

print("=" * 70)
print("  🎊 ALL TESTS COMPLETE! 🎊")
print("=" * 70)
print()
print("Notice the [LSTM], [Word-NGram], or [Char-NGram] prefix?")
print("That tells you which model generated the text!")
print()
print("LSTM = Most readable (neural network)")
print("Word-NGram = Pretty good (10-gram word model)")
print("Char-NGram = Chaotic but poetic (10-gram character model)")
print()
