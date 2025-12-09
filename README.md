# 🎵 git.symphony

> *"What if git repositories could dream?"*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**git.symphony** is a poetic git repository explorer that navigates code histories through *dreams*, *resonance*, and *entropy*. Forked from the conceptual foundations of Karpathy's rendergit, symphony doesn't just search — it **wanders**, **remembers**, and **resonates** with the patterns it discovers.

## 🌊 What Is This Madness?

Have you ever wanted to explore git repositories like you're traveling through a fever dream of commit messages? Have you ever thought, "I wish my git search tool used Markov chains and calculated the perplexity of my prompts"? 

**No?** Well, NOW YOU DO! 🎭

Symphony treats git exploration as a journey through conceptual space. It:
- 🧠 **Remembers** using SQLite databases that grow organically
- 🎲 **Wanders** through git history using Markov chains
- 📡 **Resonates** with your prompts using trigram matching
- 🌀 **Calculates** entropy, perplexity, and resonance scores
- 💾 **Dreams** in binary shards (pickled n-gram statistics)
- 🎨 **Visualizes** exploration paths as ASCII art
- 🤖 **Responds** using a CPU-only character-level language model (no PyTorch!)

## 🚀 Quick Start

```bash
# Clone this beautiful mess
git clone https://github.com/ariannamethod/git.symphony
cd git.symphony

# Install dependencies (just numpy for now!)
pip install numpy

# Enter the REPL and start dreaming
python symphony.py
```

## 🎮 Usage

Symphony runs in REPL mode. Just type what you're looking for:

```
🎵 symphony> find me transformer implementations

  ♪ Symphony is exploring...
  
  🔍 Main keyword: 'transformer'
  📊 Prompt entropy: 4.127, perplexity: 17.503
  
  💭 Generating resonance response...
  
  🌊 Symphony's Response:
  ------------------------------------------------------------------
  The transformer architecture revolutionized deep learning through
  its attention mechanism, enabling parallel processing and better
  long-range dependencies in neural networks...
  ------------------------------------------------------------------

======================================================================
  🎵 SYMPHONY'S JOURNEY 🎵
======================================================================

  User Prompt: 'find me transformer implementations'

  Metrics:
    → Resonance:  0.687 📡
    → Entropy:    4.127 🌀
    → Perplexity: 17.503 🧩

  Path Taken:
    ╔══> transformer
    ╠══> architecture
    ╠══> implementations
    ╠══> attention
    ╠══> mechanism
    ╚══> model ⭐

======================================================================

  Open repository in browser? (y/n): y
  🌐 Opened https://github.com/karpathy/nanoGPT in browser
```

See `tests/example_interaction.md` for a full session transcript!

## 🏗️ Architecture

### Core Modules

#### `symphony.py` - The Conductor
The main REPL interface and exploration engine. Features:
- 🎯 **Entropy-based keyword extraction** - finds the most informationally dense words in your prompt
- 📊 **Metric calculation** - computes resonance (trigram overlap), entropy (information density), and perplexity
- 🗺️ **Markov exploration** - uses 1-2 Markov chains to navigate through git commit histories
- 💾 **Dynamic memory** - SQLite database that grows organically, creating new tables for discovered technologies
- 🎨 **ASCII visualization** - beautiful path drawings showing how symphony found each repository
- 🔄 **Memory rotation** - automatically archives databases when they hit 2MB, keeping the old ones

#### `frequency.py` - The Dreamer
A CPU-only character-level text generator inspired by Karpathy's nanoGPT `sample.py`. Features:
- 🚫 **No PyTorch** - pure Python + NumPy, runs on CPU
- 📝 **Character-level modeling** - learns from documentation at the character level
- 🎲 **Temperature sampling** - configurable randomness (default 0.85)
- 💾 **Binary shards** - saves learned patterns as `.bin` files in the `bin/` directory
- ⚡ **Fast inference** - generates responses in milliseconds

### The Memory System

Symphony maintains a living memory in SQLite:

```sql
-- Core repository records
repositories (id, url, local_path, last_accessed, access_count, ...)

-- Exploration trails - HOW symphony found things
exploration_trails (repo_id, prompt, path_taken, resonance_score, ...)

-- Commit snapshots with discovered technologies
commit_snapshots (repo_id, commit_hash, interesting_tech, ...)

-- Dynamic technology columns added on discovery!
ALTER TABLE repositories ADD COLUMN tech_python INTEGER DEFAULT 0
ALTER TABLE repositories ADD COLUMN tech_transformer INTEGER DEFAULT 0
```

The database **grows organically** - when symphony discovers a new technology or interesting repository name, it creates a new column to track it!

### Binary Shards 🧠

The `bin/` directory stores "memory shards" - pickled n-gram statistics that represent what symphony has learned:

```
bin/
  ├── memory_shard_0001.bin
  ├── memory_shard_0002.bin
  └── memory_shard_0003.bin
```

Each shard contains:
- N-gram character transition probabilities
- Vocabulary statistics
- Character frequency distributions

Think of them as **weight checkpoints** but for a statistical model, not a neural network!

## 🎭 Key Features

### 1. Trigram-Based Search
Symphony searches git commits using **trigram matching** - breaking your prompt and commit messages into 3-character sequences and finding overlaps.

### 2. Entropy & Perplexity Metrics
Your prompts are analyzed for:
- **Entropy** - information density
- **Perplexity** - how "surprising" the text is
- **Resonance** - trigram overlap with found content

These metrics help symphony understand WHERE THE IMPORTANT WORDS ARE (yes, in caps, because it matters!).

### 3. Markov Chain Navigation
Symphony doesn't just search - it **wanders**. Using 1-2 Markov chains trained on commit messages, it generates exploration paths through conceptual space.

### 4. Memory & Forgetting
Repositories that aren't revisited gradually fade into archives. When the SQLite database hits 2MB, it rotates to a new file, keeping the old one but starting fresh. **Symphony learns to forget!**

### 5. Character-Level Response Generation
Using `frequency.py`, symphony "digests" README files and generates contextual responses. It's like Karpathy's Shakespeare generator, but for technical documentation!

```python
# From frequency.py - no PyTorch needed!
model = CharacterModel(order=4)
model.train(readme_text)
response = model.generate(seed="The", length=150, temperature=0.85)
```

### 6. ASCII Path Visualization
Every search shows you HOW symphony found what it found:

```
╔══> neural
╠══> network
╠══> training
╠══> optimization
╠══> gradient
╚══> descent ⭐
```

### 7. Browser Integration with Confirmation
Symphony asks before opening your browser (y/n prompts, like a proper terminal tool).

## 🧪 Testing

Run the test suite:

```bash
python tests/test_symphony_basic.py
```

Tests cover:
- ✅ Entropy calculation
- ✅ Perplexity scoring
- ✅ Resonance (trigram matching)
- ✅ Keyword extraction
- ✅ Markov chain generation
- ✅ Frequency text generation
- ✅ SQLite memory operations

Check `tests/example_interaction.md` for a spectacular example session!

## 🎨 Example Output

```
🎵 symphony> looking for character level language models

  🔍 Main keyword: 'character'
  📊 Prompt entropy: 4.301, perplexity: 19.685

  🌊 Symphony's Response:
  ------------------------------------------------------------------
  Character-level modeling operates at the finest granularity of
  text, treating each individual character as a token. This approach
  has unique advantages: no tokenization needed, can handle any
  text, and generates at character frequency...
  ------------------------------------------------------------------

======================================================================
  🎵 SYMPHONY'S JOURNEY 🎵
======================================================================

  User Prompt: 'looking for character level language models'

  Metrics:
    → Resonance:  0.542 📡
    → Entropy:    4.301 🌀
    → Perplexity: 19.685 🧩

  Path Taken:
    ╔══> character
    ╠══> level
    ╠══> language
    ╠══> models
    ╠══> text
    ╚══> generation ⭐

======================================================================
```

## 🤔 Why Does This Exist?

Because sometimes you need to search git repositories **POETICALLY**. Because entropy and perplexity matter. Because Markov chains are beautiful. Because code should dream.

Also, it's really fun to watch symphony wander through commit histories and generate slightly surreal but grammatically correct responses about the code it finds.

## 🔮 Future Plans

This is **beta v1**. Future versions will include:
- 🦙 **Miniature Llama integration** - even more surreal explorations
- 🌐 **GitHub API integration** - search actual remote repositories
- 🧬 **Multi-chain Markov** - more complex wandering patterns
- 🎪 **Visualization modes** - graph-based path displays
- 🎯 **Smart caching** - remember successful exploration patterns

## 🛠️ Technical Details

### Dependencies
- Python 3.8+
- NumPy (for probability distributions)
- SQLite3 (built-in)
- Standard library (subprocess, pathlib, etc.)

**No PyTorch. No TensorFlow. No heavy ML frameworks.**

Just pure, beautiful, slightly unhinged Python code.

### File Structure

```
git.symphony/
├── symphony.py          # Main REPL and exploration engine
├── frequency.py         # Character-level text generator (CPU-only)
├── bin/                 # Binary shards (memory weights) - gitignored
│   └── memory_shard_*.bin
├── tests/
│   ├── test_symphony_basic.py
│   └── example_interaction.md
├── *.db                 # SQLite databases - gitignored
└── README.md            # You are here! 👋
```

### How It Works (The Technical Poetry)

1. **User enters prompt** → Symphony calculates entropy & perplexity
2. **Keyword extraction** → Identifies main concept using information theory
3. **Git search** → Finds commits using trigram resonance matching
4. **Markov wandering** → Generates exploration path through commit messages
5. **README discovery** → Locates and loads the best matching documentation
6. **Frequency digestion** → Character model "eats" the README
7. **Response generation** → Produces contextual, poetic output
8. **Memory recording** → Saves everything to SQLite with metrics
9. **Path visualization** → Draws ASCII art of the journey
10. **Browser launch** → Opens repository with user confirmation

All of this happens in **seconds** on a CPU. No GPU needed. No cloud API calls. Just local, poetic, slightly mad exploration.

## 🎪 Contributing

This is an art project as much as a tool. Contributions welcome, especially:
- 🎨 More ASCII art styles
- 🎲 Alternative Markov chain strategies  
- 📊 New metric calculations
- 🎭 Surreal response templates
- 🌈 Color themes for terminal output

Keep it weird. Keep it wonderful.

## 📜 License

MIT License - go forth and make git repositories dream!

## 🙏 Acknowledgments

Forked conceptually from Karpathy's rendergit. The character-level generation in `frequency.py` is inspired by the approach in nanoGPT's `sample.py`, but implemented without PyTorch for pure CPU speed.

## 💬 Final Words

> *"Symphony doesn't search. It wanders. It dreams. It resonates."*

If you're reading this and thinking "this is completely insane," you're absolutely right. But it's also kind of beautiful, isn't it? 🎵

Now go forth and let your git repositories dream through the night!

---

Made with 🎭 and 🌀 by developers who believe code should be poetic.

*P.S. - The SQLite database that grows organically? That's not a bug, that's a feature. Symphony is alive.*
