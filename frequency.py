#!/usr/bin/env python3
"""
frequency.py - A CPU-only character-level text generator inspired by Karpathy's nanoGPT sample.py

No PyTorch, no GPU needed. Just pure Python magic for generating poetic technical responses.
Uses character-level modeling to "digest" documentation and produce grammatically coherent,
slightly surreal responses.
"""

import random
import pickle
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


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


class FrequencyEngine:
    """
    Main frequency engine that manages models and generation.
    Stores model weights as binary shards in bin/ directory.
    """
    
    def __init__(self, bin_dir: str = "bin"):
        self.bin_dir = Path(bin_dir)
        self.bin_dir.mkdir(exist_ok=True)
        self.model = CharacterModel(order=4)
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
        Digest text into the model, building up memories.
        
        Args:
            text: Text to digest (usually README content)
        """
        # Clean and prepare text
        text = text[:5000]  # Limit to first 5000 chars for speed
        
        # Train model on this text
        self.model.train(text)
        
        # Periodically save shards
        if self.model.total_chars > 50000:  # Every ~50k chars
            self.save_shard()
            # Reset counters but keep learned patterns
            self.model.total_chars = 0
    
    def generate_response(self, text: str, seed: str = "", max_length: int = 150) -> str:
        """
        Generate a response after digesting text.
        
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
            # Extract a random interesting phrase from the text
            sentences = text.split('.')
            if sentences:
                seed = random.choice(sentences)[:20].strip()
                if not seed:
                    seed = "The code "
        
        # Generate response
        raw_response = self.model.generate(seed, length=max_length, temperature=0.85)
        
        # Post-process: try to end at a sentence boundary
        response = self._clean_response(raw_response, max_length)
        
        return response
    
    def _clean_response(self, text: str, max_length: int) -> str:
        """Clean up generated response to make it more presentable."""
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
