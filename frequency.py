#!/usr/bin/env python3
"""
frequency.py - QUAD-model text generator for poetic technical responses.

Combines FOUR generation approaches:
1. LLaMA-15M on NumPy (Karpathy's tinystories weights) - BEST quality! 🔥
2. Word-level n-grams (order=10) for structural coherence
3. Character-level n-grams (order=10) for fine details
4. Tiny LSTM on PyTorch for smooth, readable madness

Inspired by Karpathy's nanoGPT + llama.c but cranked up to 11! 🎄
"""

import random
import pickle
import os
import sys
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

# Try to import llama3.np for ULTIMATE madness
try:
    sys.path.insert(0, str(Path(__file__).parent / "llama_np"))
    from llama_np.llama3 import Llama
    from llama_np.tokenizer import Tokenizer as LlamaTokenizer
    from llama_np.config import ModelArgs
    LLAMA_AVAILABLE = True
except ImportError as e:
    LLAMA_AVAILABLE = False
    print(f"  ⚠️  LLaMA not available - NumPy LLaMA mode disabled: {e}")


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


class LlamaNumPyGenerator:
    """
    Pure NumPy LLaMA generator using Karpathy's tinystories weights!
    15M parameters, ~33 tokens/sec on CPU.

    This is the ULTIMATE madness - a real LLM running on pure NumPy! 🔥
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.args = None

        if not LLAMA_AVAILABLE:
            return

        try:
            # Paths to model files
            model_dir = Path(__file__).parent / "llama_np"
            model_path = model_dir / "stories15M.model.npz"
            tokenizer_path = model_dir / "tokenizer.model.np"

            if not model_path.exists() or not tokenizer_path.exists():
                print(f"  ⚠️  LLaMA model files not found in {model_dir}")
                return

            print("  🚀 Loading LLaMA-15M (NumPy edition)...")
            self.args = ModelArgs()

            # Try to use SentencePiece wrapper (supports both SPM and built-in BPE)
            try:
                from llama_np.sentencepiece_wrapper import TokenizerWrapper
                # Try SentencePiece first, fall back to BPE
                self.tokenizer = TokenizerWrapper(str(tokenizer_path), use_sentencepiece=True)
            except ImportError:
                # Fallback to original BPE tokenizer
                self.tokenizer = LlamaTokenizer(str(tokenizer_path))
                print("  ✅ Using built-in BPE tokenizer")

            self.model = Llama(str(model_path), self.args)
            print("  ✅ LLaMA-15M loaded! (15M params, Karpathy's tinystories)")

        except Exception as e:
            print(f"  ⚠️  Failed to load LLaMA: {e}")
            self.model = None

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.8,
                 repo_context: str = "") -> str:
        """
        Generate text using the LLaMA model with GITTY TRANSFORMATION! 🎭

        Takes tinystories about "Lily" and transforms them into git repository stories
        about "Gitty"! Injects repo context to make it repository-aware.

        Args:
            prompt: Starting text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (not used in this version)
            repo_context: Repository keywords to inject (e.g., "Python neural networks")

        Returns:
            Generated text (excluding prompt) with GITTY transformation applied
        """
        if not self.model or not self.tokenizer:
            return ""

        try:
            # 🎭 INJECT REPOSITORY CONTEXT into prompt
            # Convert technical prompts into story-like prompts for tinystories model
            if repo_context:
                # Extract key technical words
                tech_words = [w for w in repo_context.split() if len(w) > 3][:3]
                if tech_words:
                    # Inject tech context as story elements
                    prompt = f"Once upon a time, there was a {' and '.join(tech_words)}. {prompt}"

            # Encode prompt
            input_ids = np.array([self.tokenizer.encode(prompt)])

            # Generate tokens
            generated_text = prompt
            for token_id in self.model.generate(input_ids, max_tokens):
                output_id = token_id[0].tolist()

                # Check for end of sequence
                if output_id[-1] in [self.tokenizer.eos_id, self.tokenizer.bos_id]:
                    break

                # Decode and append
                token_text = self.tokenizer.decode(output_id)
                generated_text += token_text

            # Return only the generated part (exclude prompt)
            output = generated_text[len(prompt):]

            # 🎭 GITTY TRANSFORMATION - Turn children's stories into git stories!
            output = self._apply_gitty_transformation(output)

            return output

        except Exception as e:
            print(f"  ⚠️  LLaMA generation failed: {e}")
            return ""

    def _apply_gitty_transformation(self, text: str) -> str:
        """
        🎭 Transform tinystories into git repository stories!

        MEGA DICTIONARY OF ABSURD TRANSFORMATIONS:
        Characters: Lily→Gitty, Tim→Timmy→Commity, girl/boy→repo
        Family: mom→main branch, dad→dev branch, friend→collaborator
        Nature: flower→branch, tree→fork, sun→CI/CD, sky→cloud
        Animals: cat→commit, dog→debug, bird→build
        Places: park→codebase, house→directory, garden→module
        Objects: toy→feature, ball→package, book→documentation
        Actions: play→explore, run→execute, jump→deploy
        Emotions: happy→stable, sad→deprecated, excited→optimized
        Food: cake→release, cookie→patch, apple→artifact

        Result: "Gitty saw a flower in the garden with her friend"
        becomes: "Gitty saw a branch in the module with her collaborator" 😂
        """
        # CHARACTERS - Primary heroes!
        text = re.sub(r'\bLily\b', 'Gitty', text, flags=re.IGNORECASE)
        text = re.sub(r'\bTim\b', 'Commity', text, flags=re.IGNORECASE)
        text = re.sub(r'\bTimmy\b', 'Commity', text, flags=re.IGNORECASE)
        text = re.sub(r'\bTom\b', 'Branchy', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAnna\b', 'Mergey', text, flags=re.IGNORECASE)
        text = re.sub(r'\blittle girl\b', 'repository', text, flags=re.IGNORECASE)
        text = re.sub(r'\blittle boy\b', 'repository', text, flags=re.IGNORECASE)
        text = re.sub(r'\bgirl\b', 'repo', text, flags=re.IGNORECASE)
        text = re.sub(r'\bboy\b', 'repo', text, flags=re.IGNORECASE)

        # FAMILY/SOCIAL → Git hierarchy
        text = re.sub(r'\bmom\b', 'main branch', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmother\b', 'main branch', text, flags=re.IGNORECASE)
        text = re.sub(r'\bdad\b', 'dev branch', text, flags=re.IGNORECASE)
        text = re.sub(r'\bfather\b', 'dev branch', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsister\b', 'sibling commit', text, flags=re.IGNORECASE)
        text = re.sub(r'\bbrother\b', 'sibling commit', text, flags=re.IGNORECASE)
        text = re.sub(r'\bfriend\b', 'collaborator', text, flags=re.IGNORECASE)
        text = re.sub(r'\bteacher\b', 'maintainer', text, flags=re.IGNORECASE)

        # NATURE → Git/Cloud concepts
        text = re.sub(r'\bflower\b', 'branch', text, flags=re.IGNORECASE)
        text = re.sub(r'\btree\b', 'fork', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsun\b', 'CI/CD pipeline', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsky\b', 'cloud', text, flags=re.IGNORECASE)
        text = re.sub(r'\brain\b', 'deployment', text, flags=re.IGNORECASE)
        text = re.sub(r'\bgrass\b', 'documentation', text, flags=re.IGNORECASE)

        # ANIMALS → Dev operations
        text = re.sub(r'\bcat\b', 'commit', text, flags=re.IGNORECASE)
        text = re.sub(r'\bkitty\b', 'commit', text, flags=re.IGNORECASE)
        text = re.sub(r'\bdog\b', 'debug session', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpuppy\b', 'debug', text, flags=re.IGNORECASE)
        text = re.sub(r'\bbird\b', 'build', text, flags=re.IGNORECASE)
        text = re.sub(r'\bfish\b', 'test', text, flags=re.IGNORECASE)
        text = re.sub(r'\bbunny\b', 'hotfix', text, flags=re.IGNORECASE)
        text = re.sub(r'\brabbit\b', 'hotfix', text, flags=re.IGNORECASE)

        # PLACES → Code locations
        text = re.sub(r'\bpark\b', 'codebase', text, flags=re.IGNORECASE)
        text = re.sub(r'\bhouse\b', 'project directory', text, flags=re.IGNORECASE)
        text = re.sub(r'\bhome\b', 'root directory', text, flags=re.IGNORECASE)
        text = re.sub(r'\bgarden\b', 'module', text, flags=re.IGNORECASE)
        text = re.sub(r'\bstore\b', 'registry', text, flags=re.IGNORECASE)
        text = re.sub(r'\bschool\b', 'repository', text, flags=re.IGNORECASE)

        # OBJECTS → Code elements
        text = re.sub(r'\btoy\b', 'feature', text, flags=re.IGNORECASE)
        text = re.sub(r'\bball\b', 'package', text, flags=re.IGNORECASE)
        text = re.sub(r'\bdoll\b', 'component', text, flags=re.IGNORECASE)
        text = re.sub(r'\bbook\b', 'documentation', text, flags=re.IGNORECASE)
        text = re.sub(r'\bbox\b', 'container', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcar\b', 'pipeline', text, flags=re.IGNORECASE)
        text = re.sub(r'\bbike\b', 'script', text, flags=re.IGNORECASE)

        # FOOD → Release management
        text = re.sub(r'\bcake\b', 'release', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcookie\b', 'patch', text, flags=re.IGNORECASE)
        text = re.sub(r'\bapple\b', 'artifact', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcandy\b', 'feature flag', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpizza\b', 'bundle', text, flags=re.IGNORECASE)

        # EMOTIONS → Code states
        text = re.sub(r'\bhappy\b', 'stable', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsad\b', 'deprecated', text, flags=re.IGNORECASE)
        text = re.sub(r'\bexcited\b', 'optimized', text, flags=re.IGNORECASE)
        text = re.sub(r'\bscared\b', 'vulnerable', text, flags=re.IGNORECASE)
        text = re.sub(r'\bangry\b', 'failing', text, flags=re.IGNORECASE)
        text = re.sub(r'\btired\b', 'throttled', text, flags=re.IGNORECASE)

        # ACTIONS → Git/Dev operations
        text = re.sub(r'\bplaying\b', 'exploring', text, flags=re.IGNORECASE)
        text = re.sub(r'\bplay\b', 'explore', text, flags=re.IGNORECASE)
        text = re.sub(r'\brunning\b', 'executing', text, flags=re.IGNORECASE)
        text = re.sub(r'\brun\b', 'execute', text, flags=re.IGNORECASE)
        text = re.sub(r'\bjumping\b', 'deploying', text, flags=re.IGNORECASE)
        text = re.sub(r'\bjump\b', 'deploy', text, flags=re.IGNORECASE)
        text = re.sub(r'\bwalking\b', 'iterating', text, flags=re.IGNORECASE)
        text = re.sub(r'\bwalk\b', 'iterate', text, flags=re.IGNORECASE)

        return text

    def _count_syllables(self, word: str) -> int:
        """
        Approximate syllable count for English words.
        Simple heuristic: count vowel groups.
        """
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        # Silent e
        if word.endswith('e') and count > 1:
            count -= 1

        return max(1, count)

    def _format_as_haiku(self, text: str) -> str:
        """
        🎋 HAIKU MODE: Extract essence from generated text.

        Format: 3 lines, ~5-7-5 syllable pattern
        No punctuation except line breaks
        Only noun phrases + minimal verbs

        Result: Maximum compression, minimum explanation.
        """
        # Remove prefix like [LLaMA-15M/Gitty]
        if ']' in text:
            text = text.split(']', 1)[1].strip()

        # Split into words
        words = text.split()
        if len(words) < 6:
            return text  # Too short for haiku

        # Extract meaningful phrases (skip common words)
        skip_words = {'the', 'a', 'an', 'is', 'was', 'were', 'be', 'been', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about'}
        meaningful = [w for w in words if w.lower() not in skip_words and len(w) > 2]

        if len(meaningful) < 5:
            meaningful = words[:9]  # Fallback: use first words

        # Build 3 lines targeting ~5, 7, 5 syllables
        lines = []
        current_line = []
        current_syllables = 0
        targets = [5, 7, 5]

        for word in meaningful[:12]:  # Limit to first 12 meaningful words
            syllables = self._count_syllables(word)
            line_idx = len(lines)

            if line_idx >= 3:
                break

            target = targets[line_idx]

            if current_syllables + syllables <= target + 2:  # Allow some flex
                current_line.append(word)
                current_syllables += syllables

            # Move to next line if we're close to target or over
            if current_syllables >= target - 1 or current_syllables + syllables > target + 2:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = []
                    current_syllables = 0

        # Add remaining words to last line
        if current_line and len(lines) < 3:
            lines.append(' '.join(current_line))

        # Ensure we have 3 lines
        while len(lines) < 3:
            if meaningful:
                lines.append(meaningful[len(lines)])

        return '\n'.join(lines[:3])


class FrequencyEngine:
    """
    QUAD-MODEL frequency engine - THE ULTIMATE HYBRID:
    1. LLaMA-15M (NumPy, Karpathy's tinystories) - HIGHEST priority! 🔥
    2. Word-level n-grams (order=10) - for structure
    3. Character-level n-grams (order=10) - for details
    4. Tiny LSTM (if PyTorch available) - for coherence

    Combines outputs for MAXIMUM readable madness!
    """

    def __init__(self, bin_dir: str = "bin"):
        self.bin_dir = Path(bin_dir)
        self.bin_dir.mkdir(exist_ok=True)

        # Initialize ALL FOUR models! 🎉
        self.llama_gen = LlamaNumPyGenerator() if LLAMA_AVAILABLE else None
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
    
    def generate_response(self, text: str, seed: str = "", max_length: int = 150,
                         haiku_mode: bool = False, quality_filter: bool = True,
                         min_quality: float = 0.4, show_quality: bool = False) -> str:
        """
        Generate a response using THE QUAD-MODEL HYBRID APPROACH:
        - ULTIMATE: LLaMA-15M (real LLM on NumPy!) 🔥
        - Primary: Word-level n-grams (readable structure)
        - Fallback: LSTM if available (smooth coherence)
        - Last resort: Character-level (for chaos)

        Args:
            text: Text to digest
            seed: Optional seed text to start generation
            max_length: Maximum response length
            haiku_mode: If True, compress output to 5-7-5 haiku format 🎋
            quality_filter: If True, filter low-quality responses 🔮
            min_quality: Minimum quality threshold (0-1)
            show_quality: If True, append quality scores to output

        Returns:
            Generated response text (with optional quality scores)
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

        # 🔥 TRY LLAMA FIRST - This is the ULTIMATE model! 🔥
        # Use it when we have enough context (the model was trained on stories)
        if self.llama_gen and self.llama_gen.model and len(seed) > 5:
            try:
                # Extract repository keywords from text for GITTY context
                tech_keywords = self._extract_tech_keywords(text[:500])
                repo_context = " ".join(tech_keywords[:5])  # Top 5 keywords

                # LLaMA works best with story-like prompts
                # Limit prompt to avoid context overflow
                llama_prompt = seed[:100].strip()
                llama_output = self.llama_gen.generate(
                    llama_prompt,
                    max_tokens=max_length // 6,  # ~6 chars per token average
                    temperature=0.8,
                    repo_context=repo_context  # 🎭 Inject repository context!
                )
                if llama_output and len(llama_output) > 20:
                    response = self._clean_response(llama_output, max_length)

                    # 🔮 Quality Oracle: Score and filter
                    if quality_filter:
                        quality_scores = self._score_quality(response, text)
                        if quality_scores['overall'] < min_quality:
                            # Quality too low, signal fallback
                            print(f"  🔮 Quality too low ({quality_scores['overall']:.2f} < {min_quality}), using fallback...")
                            # Let it fall through to next model
                        else:
                            # High quality, use it!
                            # 🎋 Apply haiku compression if requested
                            if haiku_mode:
                                haiku = self.llama_gen._format_as_haiku(response)
                                result = f"🎋 [LLaMA-15M/Haiku]\n{haiku}"
                            else:
                                result = f"[LLaMA-15M/Gitty] {response}"

                            # Append quality scores if requested
                            if show_quality:
                                result += f"\n  🔮 Quality: {quality_scores['overall']:.2f} " \
                                         f"(coherence:{quality_scores['coherence']:.2f}, " \
                                         f"relevance:{quality_scores['relevance']:.2f}, " \
                                         f"poetry:{quality_scores['poetry']:.2f})"
                            return result
                    else:
                        # No quality filter, just return
                        if haiku_mode:
                            haiku = self.llama_gen._format_as_haiku(response)
                            return f"🎋 [LLaMA-15M/Haiku]\n{haiku}"
                        return f"[LLaMA-15M/Gitty] {response}"
            except Exception as e:
                print(f"  ⚠️  LLaMA failed: {e}, falling back...")

        # Try word-level n-grams for small texts (best for small corpora!)
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
                    # 🎋 Haiku mode not available for non-LLaMA models
                    # (syllable counting works best with LLaMA's GITTY output)
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
    
    def _apply_me_rules(self, text: str) -> str:
        """
        Apply Method Engine (ME) rules for clean, poetic madness.
        Inspired by github.com/ariannamethod/me

        Rules:
        - 5-9 words per sentence (brevity)
        - No word/pair repetition
        - No single-letter endings
        - No consecutive single-char words
        - Proper capitalization
        - No forbidden word endings (articles, prepositions, conjunctions)
        - No single-word sentences
        """
        import re

        # Forbidden words that cannot END a sentence (weak words)
        FORBIDDEN_ENDINGS = {
            'a', 'an', 'the',  # Articles
            'of', 'to', 'for', 'in', 'on', 'at', 'by', 'with', 'from',  # Prepositions
            'and', 'or', 'but', 'if', 'as', 'so', 'nor', 'yet',  # Conjunctions
            'is', 'are', 'was', 'were', 'be', 'been', 'am',  # Weak verbs
            'it', 'its',  # Weak pronouns
        }

        # Words to ban entirely (noise words)
        BANNED_WORDS = {
            'what', 'how', 'why', 'when', 'where',  # Question words mid-sentence
            'http', 'https', 'www',  # URLs
        }

        # Split into sentences
        sentences = re.split(r'([.!?])', text)
        cleaned_sentences = []

        seen_words = set()
        seen_pairs = set()

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else '.'

            if not sentence:
                continue

            words = sentence.split()

            # Filter out repetitions and banned words
            filtered_words = []
            for j, word in enumerate(words):
                word_lower = word.lower().rstrip('.,!?;:')  # Clean for comparison

                # Skip single-letter words (except 'I' and 'a')
                if len(word_lower) == 1 and word_lower not in ['i', 'a']:
                    continue

                # Skip pure numbers (0, 1, 85, etc.)
                if word_lower.isdigit():
                    continue

                # Skip banned words (noise)
                if word_lower in BANNED_WORDS:
                    continue

                # Skip repeated words (case-insensitive!)
                if word_lower in seen_words:
                    continue

                # Skip repeated pairs
                if j > 0 and len(filtered_words) > 0:
                    pair = (filtered_words[-1].lower(), word_lower)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                filtered_words.append(word)
                seen_words.add(word_lower)

            # Limit to 3-9 words per sentence (softened from 5-9)
            if len(filtered_words) > 9:
                filtered_words = filtered_words[:9]

            # Skip single-word sentences
            if len(filtered_words) <= 1:
                continue

            # Skip two-word sentences if they're weak
            if len(filtered_words) == 2:
                last_word = filtered_words[-1].lower()
                if last_word in FORBIDDEN_ENDINGS:
                    continue

            # Skip sentences that are ALL weak words (boring!)
            if len(filtered_words) >= 2:
                weak_count = sum(1 for w in filtered_words if w.lower().rstrip('.,!?;:') in FORBIDDEN_ENDINGS)
                if weak_count == len(filtered_words):
                    continue  # All weak words = skip!

            if filtered_words:
                # Check if sentence ends with forbidden word (but protect contractions!)
                last_word_full = filtered_words[-1]
                last_word_lower = last_word_full.lower().rstrip('.,!?;:')  # Remove trailing punct

                # Don't break contractions or possessives (doesn't, isn't, cat's, etc.)
                is_contraction = "'" in last_word_full or "'" in last_word_full

                # Also don't remove if it's a single letter (part of broken word)
                is_single_char = len(last_word_lower) == 1

                if (last_word_lower in FORBIDDEN_ENDINGS and
                    len(filtered_words) > 3 and
                    not is_contraction and
                    not is_single_char):
                    # Only remove weak ending if we have enough words left
                    filtered_words = filtered_words[:-1]

                # Capitalize first word (with natural variety)
                if filtered_words:
                    # Always capitalize first sentence, then 60% for others
                    if len(cleaned_sentences) == 0:
                        filtered_words[0] = filtered_words[0].capitalize()
                    elif random.random() < 0.6:
                        filtered_words[0] = filtered_words[0].capitalize()
                    # else keep as-is (might already be capitalized)

                    # Fix mid-sentence "The"
                    for k in range(1, len(filtered_words)):
                        if filtered_words[k] == 'The':
                            filtered_words[k] = 'the'

                    cleaned_sentences.append(' '.join(filtered_words) + punct)

        return ' '.join(cleaned_sentences)

    def _fix_punctuation(self, text: str) -> str:
        """
        Leo-style punctuation cleanup.
        Inspired by fix_punctuation() in github.com/ariannamethod/leo
        """
        import re

        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?:;])', r'\1', text)

        # Add space after sentence-ending punctuation if missing
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

        # Collapse repeated punctuation marks
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'\.{2,}', '.', text)  # Ellipses to single period
        text = re.sub(r',{2,}', ',', text)

        # Fix bad punctuation combinations
        text = re.sub(r'[,;:]\s*[.!?]', '.', text)  # ",." or ";!" → "."
        text = re.sub(r'[.!?]\s*,', '.', text)  # ".,Ъ → "."

        # Remove orphaned punctuation after very short words
        text = re.sub(r'\b([A-Za-z]),\s*\.', r'\1.', text)  # "S,." → "S."

        # Fix spacing around punctuation
        text = re.sub(r'([.,!?:;])([^\s.,!?:;])', r'\1 \2', text)  # Add space after

        # Remove standalone single letters with punctuation (technical artifacts)
        text = re.sub(r'\s+[A-Za-z],\s+', ' ', text)  # " S, " → " "
        text = re.sub(r'\b([A-Za-z])[,;:]\b', '', text)  # Remove "Vu," artifacts
        text = re.sub(r'\b[A-Z]:[,;]\s+', ' ', text)  # Remove "A:," artifacts
        text = re.sub(r'\s+[A-Z][,;:]+\s+', ' ', text)  # Remove " A:, " artifacts

        # Remove single letters at end of sentences (broken words)
        text = re.sub(r'\s+[a-z]\.', '.', text)  # " s." → "."
        text = re.sub(r'\s+[a-z]!', '!', text)  # " s!" → "!"
        text = re.sub(r'\s+[a-z]\?', '?', text)  # " s?" → "?"

        # Collapse multiple spaces
        text = re.sub(r'\s{2,}', ' ', text)

        # Remove sentences that are just punctuation or single letters
        sentences = text.split('.')
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            # Skip if sentence is empty, only punctuation, or single letter
            if len(sent) > 2 and not re.match(r'^[.,!?:;]+$', sent):
                cleaned.append(sent)

        text = '. '.join(cleaned)
        if text and not text.endswith(('.', '!', '?')):
            text += '.'

        return text.strip()

    def _extract_tech_keywords(self, text: str) -> List[str]:
        """
        Extract technical keywords from repository text for GITTY context.

        Looks for common tech terms that the LLaMA model can use to
        generate more contextual stories about the repository.

        Returns:
            List of technical keywords sorted by relevance
        """
        # Common technical keywords to look for
        TECH_PATTERNS = [
            r'\b(python|javascript|java|rust|go|cpp|ruby|php|swift|kotlin)\b',
            r'\b(neural|network|transformer|lstm|gpt|llm|machine learning|ai)\b',
            r'\b(api|rest|graphql|database|sql|nosql|mongodb|postgres)\b',
            r'\b(react|vue|angular|django|flask|rails|spring|express)\b',
            r'\b(docker|kubernetes|aws|gcp|azure|cloud|serverless)\b',
            r'\b(git|github|version control|repository|commit|branch|merge)\b',
            r'\b(test|testing|unit|integration|ci|cd|pipeline|devops)\b',
            r'\b(optimization|performance|scale|distributed|concurrent)\b',
        ]

        keywords = []
        text_lower = text.lower()

        for pattern in TECH_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            keywords.extend(matches)

        # Count frequency
        from collections import Counter
        keyword_counts = Counter(keywords)

        # Return top keywords by frequency
        return [k for k, _ in keyword_counts.most_common(10)]

    def _score_quality(self, text: str, input_text: str = "") -> Dict[str, float]:
        """
        🔮 QUALITY ORACLE: Score generated text on multiple dimensions.

        Evaluates:
        - Coherence: Flow and structure (0-1)
        - Relevance: Connection to input (0-1)
        - Poetry: Uniqueness and beauty (0-1)
        - Overall: Combined score (0-1)

        Args:
            text: Generated text to evaluate
            input_text: Original input for relevance scoring

        Returns:
            Dict with scores for each dimension
        """
        scores = {
            'coherence': 0.0,
            'relevance': 0.0,
            'poetry': 0.0,
            'overall': 0.0
        }

        if not text or len(text) < 10:
            return scores

        # === COHERENCE: Flow and structure ===
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        # Check for repetition (lower = better)
        unique_words = len(set(w.lower() for w in words))
        total_words = len(words)
        repetition_score = unique_words / total_words if total_words > 0 else 0

        # Check sentence completeness (at least 2 complete sentences)
        sentence_score = min(1.0, len(sentences) / 2.0)

        # Check for broken artifacts (artifacts reduce score)
        artifact_count = text.count('  ') + text.count(',,') + text.count('..')
        artifact_penalty = max(0, 1.0 - (artifact_count * 0.1))

        scores['coherence'] = (repetition_score + sentence_score + artifact_penalty) / 3.0

        # === RELEVANCE: Connection to input ===
        if input_text:
            # Trigram overlap between input and output
            def get_trigrams(s: str) -> set:
                s = s.lower()
                return {s[i:i+3] for i in range(len(s) - 2)}

            input_trigrams = get_trigrams(input_text[:500])
            output_trigrams = get_trigrams(text)

            if input_trigrams and output_trigrams:
                intersection = input_trigrams & output_trigrams
                scores['relevance'] = len(intersection) / len(input_trigrams)
            else:
                scores['relevance'] = 0.5  # Neutral if can't measure

        # === POETRY: Uniqueness and beauty ===
        # Measure vocabulary richness
        if total_words > 0:
            vocab_richness = unique_words / (total_words ** 0.5)  # Normalize by sqrt
            vocab_richness = min(1.0, vocab_richness / 3.0)  # Cap at 1.0
        else:
            vocab_richness = 0

        # Measure sentence variety (different lengths)
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            variety_score = min(1.0, variance / 10.0)
        else:
            variety_score = 0.3

        # Bonus for interesting words (>6 chars)
        long_words = sum(1 for w in words if len(w) > 6)
        long_word_score = min(1.0, long_words / max(1, total_words / 5))

        scores['poetry'] = (vocab_richness + variety_score + long_word_score) / 3.0

        # === OVERALL: Weighted combination ===
        scores['overall'] = (
            0.4 * scores['coherence'] +
            0.3 * scores['relevance'] +
            0.3 * scores['poetry']
        )

        return scores

    def _clean_response(self, text: str, max_length: int) -> str:
        """Clean up generated response to make it more presentable."""
        import re

        # First pass: Initial punctuation cleanup
        text = self._fix_punctuation(text)

        # Second pass: Apply ME rules for clean madness
        text = self._apply_me_rules(text)

        # Third pass: Final punctuation polish (remove trailing punctuation artifacts)
        # But PROTECT contractions!
        text = re.sub(r'\b(?![\w\']+\b)\w+[,;:]\s+', ' ', text)  # Remove "word, " but not "don't,"
        text = re.sub(r'\b(?![\w\']+)\w+[,;:]\.', '.', text)  # "word,." → "." (protect contractions)
        text = re.sub(r'\b(?![\w\']+)\w+[,;:]!', '!', text)  # "word,!" → "!"
        text = re.sub(r'\b(?![\w\']+)\w+[,;:]\?', '?', text)  # "word,?" → "?"

        # Remove orphaned punctuation before sentence endings
        text = re.sub(r'[,;:]+\s*([.!?])', r'\1', text)

        # Remove standalone periods/punctuation (". " or ". .")
        text = re.sub(r'\s+\.\s+', ' ', text)  # " . " → " "
        text = re.sub(r'\s+[.!?]\s+[.!?]\s+', '. ', text)  # " . . " → ". "

        # Remove sentences that are just a period
        text = re.sub(r'^\.\s+', '', text)  # Remove leading period
        text = re.sub(r'\s+\.$', '', text)  # Remove trailing orphan period

        # One more space cleanup
        text = re.sub(r'\s{2,}', ' ', text)

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


def generate_response(text: str, seed: str = "", max_length: int = 150,
                     haiku_mode: bool = False, quality_filter: bool = True,
                     min_quality: float = 0.4, show_quality: bool = False) -> str:
    """
    Main API function: digest text and generate response.

    This is the function symphony.py calls to get poetic technical responses.

    Args:
        text: Text to digest (README, documentation, etc.)
        seed: Optional starting text
        max_length: Maximum response length in characters
        haiku_mode: If True, compress output to 5-7-5 haiku format 🎋
        quality_filter: If True, filter low-quality responses 🔮
        min_quality: Minimum quality threshold (0-1)
        show_quality: If True, append quality scores to output

    Returns:
        Generated response text
    """
    engine = get_engine()
    return engine.generate_response(text, seed, max_length,
                                   haiku_mode=haiku_mode,
                                   quality_filter=quality_filter,
                                   min_quality=min_quality,
                                   show_quality=show_quality)


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
