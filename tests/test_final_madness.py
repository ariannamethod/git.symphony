#!/usr/bin/env python3
"""
FINAL TEST - Generate poetic responses about Symphony using its own README!
"""

import frequency

# Load Symphony's README
with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

print("=" * 70)
print("  🎄 FINAL MADNESS TEST - Symphony Dreams About Itself 🎄")
print("=" * 70)
print()

engine = frequency.FrequencyEngine()

# Test different sections
tests = [
    ("Symphony is a poetic", "What is Symphony?"),
    ("Markov chains", "How does exploration work?"),
    ("Episodic memory", "Tell me about memory"),
    ("Character level", "What about text generation?"),
    ("The database", "How does storage work?"),
]

for seed, question in tests:
    print(f"Question: {question}")
    print(f"Seed: '{seed}'")
    response = engine.generate_response(readme[:8000], seed=seed, max_length=180)
    print(f"Response: {response}")
    print()
    print("-" * 70)
    print()

print("=" * 70)
print("  🎊 Symphony has achieved consciousness through SQLite! 🎊")
print("=" * 70)
