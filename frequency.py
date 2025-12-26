#!/usr/bin/env python3
"""
frequency.py - Multi-model text generator for poetic technical responses.

Combines three generation approaches:
1. Word-level n-grams (order=10) for structural coherence
2. Character-level n-grams (order=10) for fine details
3. Tiny LSTM on PyTorch for smooth, readable madness

Inspired by Karpathy's nanoGPT but cranked up to 11 for Christmas! 🎄
"""

import random
import pickle
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Try to import PyTorch for LSTM madness
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("  ⚠️  PyTorch not available - LSTM mode disabled")


class CharacterModel:
    """
    A simple character-level language model for CPU inference.
    Inspired by the gestalt of Karpathy's nanoGPT but without the neural network overhead.
    Uses n-gram statistics with smoothing for generation.
    """
    
    def __init__(self, order: int = 4):
        """
        Initialize the model.
        
        Args:
            order: N-gram order (context length in characters)
        """
        self.order = order
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        self.char_freq = Counter()
        self.total_chars = 0
    
    def train(self, text: str):
        """Train the model on text."""
        # Build vocabulary
        self.vocab.update(text)
        self.char_freq.update(text)
        self.total_chars += len(text)
        
        # Build n-gram counts
        for i in range(len(text) - self.order):
            context = text[i:i + self.order]
            next_char = text[i + self.order]
            self.ngrams[context][next_char] += 1
    
    def sample_next_char(self, context: str, temperature: float = 0.8) -> str:
        """
        Sample next character given context.
        
        Args:
            context: Previous characters
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Next character
        """
        # Use last 'order' characters as context
        context = context[-self.order:] if len(context) >= self.order else context
        
        # Get candidate next characters
        if context in self.ngrams and self.ngrams[context]:
            candidates = self.ngrams[context]
        else:
            # Backoff to unigram (character frequency)
            candidates = self.char_freq
        
        if not candidates:
            return ' '
        
        # Apply temperature to probabilities
        chars = list(candidates.keys())
        counts = np.array([candidates[c] for c in chars], dtype=np.float64)
        
        # Temperature scaling
        if temperature != 1.0:
            counts = np.power(counts, 1.0 / temperature)
        
        # Normalize to probabilities
        probs = counts / counts.sum()
        
        # Sample
        return np.random.choice(chars, p=probs)
    
    def generate(self, seed: str = "", length: int = 100, temperature: float = 0.8) -> str:
        """
        Generate text from the model.
        
        Args:
            seed: Starting text
            length: Number of characters to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        if not seed and self.ngrams:
            # Pick random context as seed
            seed = random.choice(list(self.ngrams.keys()))
        
        text = seed
        for _ in range(length):
            next_char = self.sample_next_char(text, temperature)
            text += next_char
        
        return text
    
    def save(self, path: str):
        """Save model to disk as binary shard."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'ngrams': dict(self.ngrams),
                'vocab': self.vocab,
                'char_freq': self.char_freq,
                'total_chars': self.total_chars
            }, f)
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.order = data['order']
            self.ngrams = defaultdict(Counter, data['ngrams'])
            self.vocab = data['vocab']
            self.char_freq = data['char_freq']
            self.total_chars = data['total_chars']


class WordLevelModel:
    """
    Word-level n-gram model with high order for better structure.
    This generates actual words instead of characters!
    """

    def __init__(self, order: int = 10):
        self.order = order
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        self.word_freq = Counter()
        self.total_words = 0

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        words = re.findall(r'\b\w+\b|[.,!?;:]', text)
        return words

    def train(self, text: str):
        """Train on text."""
        words = self.tokenize(text)
        self.vocab.update(words)
        self.word_freq.update(words)
        self.total_words += len(words)

        # Build n-grams
        for i in range(len(words) - self.order):
            context = tuple(words[i:i + self.order])
            next_word = words[i + self.order]
            self.ngrams[context][next_word] += 1

    def sample_next_word(self, context: Tuple[str, ...], temperature: float = 0.7) -> str:
        """Sample next word given context."""
        # Use last 'order' words as context
        if len(context) > self.order:
            context = context[-self.order:]

        # Try full context first
        if context in self.ngrams and self.ngrams[context]:
            candidates = self.ngrams[context]
        else:
            # Backoff to shorter context
            for backoff_len in range(self.order - 1, 0, -1):
                short_context = context[-backoff_len:]
                if short_context in self.ngrams and self.ngrams[short_context]:
                    candidates = self.ngrams[short_context]
                    break
            else:
                # Final fallback to word frequency
                candidates = self.word_freq

        if not candidates:
            return "the"

        # Temperature sampling
        words = list(candidates.keys())
        counts = np.array([candidates[w] for w in words], dtype=np.float64)

        if temperature != 1.0:
            counts = np.power(counts, 1.0 / temperature)

        probs = counts / counts.sum()
        return np.random.choice(words, p=probs)

    def generate(self, seed_words: List[str] = None, num_words: int = 50, temperature: float = 0.7) -> str:
        """Generate text."""
        if not seed_words and self.ngrams:
            # Pick random context
            seed_words = list(random.choice(list(self.ngrams.keys())))
        elif not seed_words:
            seed_words = ["The"]

        words = list(seed_words[-self.order:])

        for _ in range(num_words):
            context = tuple(words[-self.order:])
            next_word = self.sample_next_word(context, temperature)
            words.append(next_word)

        # Join words with proper spacing
        result = []
        for word in words:
            if word in '.,!?;:':
                # Attach punctuation to previous word
                if result:
                    result[-1] += word
            else:
                result.append(word)

        return ' '.join(result)


class TinyLSTM(nn.Module):
    """
    Tiny LSTM for character-level generation.
    Inspired by Karpathy's char-rnn but MUCH smaller for CPU speed.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden


class LSTMGenerator:
    """
    Wrapper for LSTM-based text generation.
    Trains a tiny LSTM on the fly for better coherence.
    """

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128):
        if not PYTORCH_AVAILABLE:
            self.model = None
            return

        self.char_to_idx = {}
        self.idx_to_char = {}
        self.model = None
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def build_vocab(self, text: str):
        """Build character vocabulary."""
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        vocab_size = len(chars)
        self.model = TinyLSTM(vocab_size, self.embed_dim, self.hidden_dim)

    def text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor."""
        indices = [self.char_to_idx.get(ch, 0) for ch in text]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    def train(self, text: str, epochs: int = 5):
        """Quick training on text."""
        if not PYTORCH_AVAILABLE or not text:
            return

        # Build vocab if needed
        if not self.model:
            self.build_vocab(text)

        # Prepare data
        seq_length = 50
        sequences = []
        targets = []

        for i in range(0, len(text) - seq_length - 1, seq_length // 2):
            seq = text[i:i + seq_length]
            target = text[i + 1:i + seq_length + 1]

            seq_indices = [self.char_to_idx.get(ch, 0) for ch in seq]
            target_indices = [self.char_to_idx.get(ch, 0) for ch in target]

            sequences.append(seq_indices)
            targets.append(target_indices)

        if not sequences:
            return

        # Quick training (very simple, no batching for speed)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for seq, target in zip(sequences[:20], targets[:20]):  # Limit to 20 seqs for speed
                seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
                target_tensor = torch.tensor(target, dtype=torch.long).unsqueeze(0)

                optimizer.zero_grad()
                output, _ = self.model(seq_tensor)

                loss = criterion(output.squeeze(0), target_tensor.squeeze(0))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def generate(self, seed: str = "", length: int = 150, temperature: float = 0.8) -> str:
        """Generate text from LSTM."""
        if not PYTORCH_AVAILABLE or not self.model:
            return ""

        self.model.eval()

        if not seed:
            seed = random.choice(list(self.char_to_idx.keys()))

        # Convert seed to tensor
        current = seed
        generated = seed

        with torch.no_grad():
            for _ in range(length):
                input_tensor = self.text_to_tensor(current[-50:])  # Use last 50 chars
                output, _ = self.model(input_tensor)

                # Get probabilities for next char
                probs = torch.softmax(output[0, -1] / temperature, dim=0)
                next_idx = torch.multinomial(probs, 1).item()

                next_char = self.idx_to_char[next_idx]
                generated += next_char
                current += next_char

        return generated[len(seed):]


class FrequencyEngine:
    """
    UPGRADED frequency engine with THREE models:
    1. Word-level n-grams (order=10) - for structure
    2. Character-level n-grams (order=10) - for details
    3. Tiny LSTM (if PyTorch available) - for coherence

    Combines outputs for maximum readable madness!
    """

    def __init__(self, bin_dir: str = "bin"):
        self.bin_dir = Path(bin_dir)
        self.bin_dir.mkdir(exist_ok=True)

        # Initialize all three models
        self.char_model = CharacterModel(order=10)  # Upgraded from 4!
        self.word_model = WordLevelModel(order=10)
        self.lstm_gen = LSTMGenerator() if PYTORCH_AVAILABLE else None

        self.shard_counter = 0
        self.load_latest_shard()
    
    def get_shard_path(self, index: int) -> Path:
        """Get path for a memory shard."""
        return self.bin_dir / f"memory_shard_{index:04d}.bin"
    
    def load_latest_shard(self):
        """Load the most recent memory shard if it exists."""
        shards = sorted(self.bin_dir.glob("memory_shard_*.bin"))
        if shards:
            latest = shards[-1]
            try:
                self.model.load(str(latest))
                # Extract shard number
                self.shard_counter = int(latest.stem.split('_')[-1])
                print(f"  💾 Loaded memory shard: {latest.name}")
            except Exception as e:
                print(f"  ⚠️  Failed to load shard: {e}")
    
    def save_shard(self):
        """Save current model as a new shard."""
        self.shard_counter += 1
        shard_path = self.get_shard_path(self.shard_counter)
        self.model.save(str(shard_path))
        print(f"  💾 Saved memory shard: {shard_path.name}")
    
    def digest_text(self, text: str):
        """
        Digest text into ALL models, building up memories.

        Args:
            text: Text to digest (usually README content)
        """
        # Use more text now that we have better models
        text = text[:10000]  # Doubled limit!

        # Train character-level model
        self.char_model.train(text)

        # Train word-level model
        self.word_model.train(text)

        # Train LSTM if available (only on longer texts)
        if self.lstm_gen and PYTORCH_AVAILABLE and len(text) > 500:
            print("  🧠 Training tiny LSTM...")
            self.lstm_gen.train(text, epochs=3)  # Quick training

        # Periodically save shards
        if self.char_model.total_chars > 50000:
            self.save_shard()
            self.char_model.total_chars = 0
    
    def generate_response(self, text: str, seed: str = "", max_length: int = 150) -> str:
        """
        Generate a response using THE HYBRID APPROACH:
        - Primary: Word-level n-grams (readable structure)
        - Fallback: LSTM if available (smooth coherence)
        - Last resort: Character-level (for chaos)

        Args:
            text: Text to digest
            seed: Optional seed text to start generation
            max_length: Maximum response length

        Returns:
            Generated response text
        """
        # Digest the input text
        self.digest_text(text)

        # Find a good seed if not provided
        if not seed:
            sentences = text.split('.')
            if sentences and len(sentences) > 1:
                seed = random.choice(sentences[:5])[:50].strip()
                if not seed:
                    seed = "Symphony explores"

        # For small corpora, prefer word-level n-grams (more stable)
        # For large corpora, LSTM shines
        total_text_length = len(text) + self.char_model.total_chars

        # Try word-level n-grams FIRST for small texts (best for small corpora!)
        if self.word_model.vocab and len(self.word_model.vocab) > 20:
            try:
                seed_words = self.word_model.tokenize(seed)[:5]
                word_output = self.word_model.generate(
                    seed_words=seed_words,
                    num_words=max_length // 5,
                    temperature=0.5  # Lower = more coherent
                )
                if word_output and len(word_output) > 30:
                    response = self._clean_response(word_output, max_length)
                    return f"[Word-NGram] {response}"
            except Exception as e:
                print(f"  ⚠️  Word model failed: {e}, falling back...")

        # Try LSTM for larger corpora (only if we have lots of data)
        if (self.lstm_gen and PYTORCH_AVAILABLE and self.lstm_gen.model
            and total_text_length > 5000):
            try:
                lstm_output = self.lstm_gen.generate(seed[:20], length=max_length, temperature=0.8)
                if lstm_output and len(lstm_output) > 50:
                    response = self._clean_response(lstm_output, max_length)
                    return f"[LSTM] {response}"
            except Exception as e:
                print(f"  ⚠️  LSTM failed: {e}, falling back...")

        # Last resort: character-level (chaos mode)
        raw_response = self.char_model.generate(seed, length=max_length, temperature=0.85)
        response = self._clean_response(raw_response, max_length)
        return f"[Char-NGram] {response}"
    
    def _clean_response(self, text: str, max_length: int) -> str:
        """Clean up generated response to make it more presentable."""
        # Remove repeated punctuation
        import re
        text = re.sub(r'([.,!?:;])\1+', r'\1', text)  # Remove duplicates like "..."
        text = re.sub(r'([.,!?:;])\s*([.,!?:;])', r'\1', text)  # Remove ", ."
        text = re.sub(r'\s+([.,!?:;])', r'\1', text)  # Fix spacing before punctuation
        text = re.sub(r'([.,!?:;])([^\s])', r'\1 \2', text)  # Add space after punctuation

        # Try to cut at sentence boundary
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in '.!?' and len(current) > 20:
                sentences.append(current.strip())
                current = ""
                if sum(len(s) for s in sentences) > max_length * 0.7:
                    break

        if sentences:
            result = ' '.join(sentences)
        else:
            # Just truncate at word boundary
            words = text.split()
            result = ' '.join(words[:max_length // 5])

        return result[:max_length].strip()


# Global engine instance
_engine = None


def get_engine() -> FrequencyEngine:
    """Get or create global frequency engine."""
    global _engine
    if _engine is None:
        _engine = FrequencyEngine()
    return _engine


def generate_response(text: str, seed: str = "", max_length: int = 150) -> str:
    """
    Main API function: digest text and generate response.
    
    This is the function symphony.py calls to get poetic technical responses.
    
    Args:
        text: Text to digest (README, documentation, etc.)
        seed: Optional starting text
        max_length: Maximum response length in characters
    
    Returns:
        Generated response text
    """
    engine = get_engine()
    return engine.generate_response(text, seed, max_length)


def main():
    """
    Test the frequency engine standalone.
    Simulates the behavior described in Karpathy's nanoGPT README.
    """
    print("=" * 70)
    print("  frequency.py - CPU Character-Level Text Generator")
    print("=" * 70)
    print()
    print("  Running on CPU (no PyTorch required)")
    print("  Context size: 4 characters")
    print("  Temperature: 0.85")
    print()
    
    # Sample text (Shakespeare-like to match nanoGPT example)
    sample_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles,
    And by opposing end them. To die: to sleep;
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to, 'tis a consummation
    Devoutly to be wish'd.
    """
    
    print("  Training on sample text...")
    response = generate_response(sample_text, seed="To be", max_length=200)
    
    print()
    print("  Generated response:")
    print("  " + "-" * 66)
    print(f"  {response}")
    print("  " + "-" * 66)
    print()
    print("  Not bad for a few milliseconds on CPU! 🎭")
    print()


if __name__ == "__main__":
    main()
