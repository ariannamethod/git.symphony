#!/usr/bin/env python3
"""
Test SentencePiece/BPE tokenization vs character-level.

Shows how BPE merges frequent character pairs into subword units.
"""

import sys
sys.path.insert(0, 'llama_np')

from llama_np.sentencepiece_wrapper import TokenizerWrapper
from llama_np.config import ModelArgs

print("=" * 70)
print("  🧪 TOKENIZATION TEST: BPE (Byte-Pair Encoding)")
print("=" * 70)
print()

# Load tokenizer
tokenizer = TokenizerWrapper("llama_np/tokenizer.model.np", use_sentencepiece=True)
print(f"Backend: {tokenizer.get_backend()}")
print()

# Test various strings
test_strings = [
    "Hello world",
    "git repository",
    "Lily was playing in the park",
    "The transformer architecture is powerful",
    "Python neural networks",
]

print("TOKENIZATION EXAMPLES:")
print("-" * 70)

for text in test_strings:
    # Encode
    tokens = tokenizer.encode(text, add_bos=False, add_eos=False)

    # Decode to see what each token represents
    token_strings = []
    for tok_id in tokens:
        single_token = tokenizer.decode([tok_id])
        token_strings.append(single_token)

    print(f"\nInput:  '{text}'")
    print(f"Tokens: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")  # First 15
    print(f"Count:  {len(tokens)} tokens")
    print(f"Pieces: {token_strings[:10]}")  # Show first 10 pieces

    # Show token/char ratio
    ratio = len(text) / max(len(tokens), 1)
    print(f"Compression: ~{ratio:.2f} chars per token")

print()
print("-" * 70)
print()

# Show character vs subword tokenization
print("CHARACTER-LEVEL vs BPE COMPARISON:")
print("-" * 70)

text = "repository"
tokens_bpe = tokenizer.encode(text, add_bos=False, add_eos=False)

print(f"\nWord: '{text}' ({len(text)} characters)")
print(f"BPE tokens: {len(tokens_bpe)} tokens")
print(f"Character-level would be: {len(text)} tokens (1 per char)")
print(f"BPE reduces token count by {100 * (1 - len(tokens_bpe)/len(text)):.1f}%")
print()

# Show actual subword pieces
pieces = [tokenizer.decode([t]) for t in tokens_bpe]
print(f"BPE breaks '{text}' into: {pieces}")
print()

print("=" * 70)
print("  💡 WHY BPE MATTERS")
print("=" * 70)
print()
print("Character-level:")
print("  - 'repository' = 10 tokens")
print("  - Each character is separate")
print("  - Model sees: r-e-p-o-s-i-t-o-r-y")
print()
print("BPE (Byte-Pair Encoding):")
print("  - 'repository' = ~6-8 tokens")
print("  - Common subwords merged: 're', 'pos', 'itory'")
print("  - Model sees meaningful chunks!")
print()
print("Benefits:")
print("  ✅ Fewer tokens = faster inference")
print("  ✅ Subwords capture morphology (re-pos-it-ory)")
print("  ✅ Better handling of rare/technical words")
print("  ✅ Vocabulary size stays manageable (~32k)")
print()
print("Our LLaMA uses BPE trained on tinystories vocabulary!")
print("=" * 70)
