# 🎵 git.symphony (aka git.haiku)

> *"What if git repositories could dream in haiku?"*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**git.symphony** (also known as **git.haiku**) is a poetic git repository explorer that navigates code histories through *dreams*, *resonance*, and *entropy*. Forked from the conceptual foundations of Karpathy's rendergit, symphony doesn't just search — it **wanders**, **remembers**, **resonates**, and **generates poetic aphorisms** about the code it discovers.

**NEW**: Now featuring **Triple-Model Hybrid Text Generation** (Word n-grams + Char n-grams + Tiny LSTM) with **Leo-style punctuation** and **ME-style brevity rules** for clean, poetic output!

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
- 🔮 **Recalls** past explorations through episodic memory (like Leo's RAG but for git adventures!)
- ⚡ **Caches** successful patterns for instant déjà vu moments

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

## 🎋 git.haiku Mode - Poetic Code Responses

Symphony's upgraded text generation creates **haiku-like aphorisms** about code. Each response is a compact, meaningful fragment - not full prose, but poetic essence!

### Real Examples from Symphony:

```
User: "What is Symphony?"
Symphony: "[Word-NGram] Symphony is a poetic."

User: "Tell me about Markov chains"
Symphony: "[Word-NGram] Markov chains: perplexity, similar.
           git PyTorch.
           The Symphony memory."

User: "How does text generation work?"
Symphony: "[Word-NGram] Character level text symphony, trigram, sql."

User: "Explain episodic memory"
Symphony: "[Word-NGram] Episodic memory."

User: "What about storage?"
Symphony: "[Word-NGram] The database: model.
           Because Forked architecture.
           It resonates."
```

### Why Aphorisms?

**Constraint breeds creativity!** Symphony uses:
- **3-9 words per sentence** (brevity)
- **No word repetition** (uniqueness)
- **No weak endings** (articles, prepositions, conjunctions banned!)
- **No noise words** (question words, numbers, URLs filtered)
- **Leo-style punctuation** (clean, minimal, precise)
- **ME-style quality** (every word counts)

Result: **Every sentence is a zen koan about code!** 🧘

```
"Markov chains: perplexity, similar."
"Character level text symphony, trigram."
"It resonates."
```

**This isn't a bug. This is art.** 🎨

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

#### `frequency.py` - The Poet (UPGRADED! 🔥)
A **Triple-Model Hybrid Engine** for generating poetic technical responses. Features:
- 🎯 **Word-Level N-Grams (order=10)** - generates actual WORDS for structural coherence
- 🔤 **Character-Level N-Grams (order=10)** - upgraded from 4! Fine-grained details
- 🧠 **Tiny LSTM on PyTorch** - 2-layer LSTM (64 embed, 128 hidden) for smooth generation
- 🎭 **Leo-style Punctuation** - clean, minimal, artifact-free (inspired by github.com/ariannamethod/leo)
- ✨ **ME-style Brevity** - 3-9 words, no repetition, no weak endings (inspired by github.com/ariannamethod/me)
- 🎋 **Aphorism Quality** - every sentence is a zen koan about code!
- 💾 **Binary shards** - saves learned patterns as `.bin` files in the `bin/` directory
- ⚡ **CPU-friendly** - LSTM trains in seconds, no GPU needed!

#### `episodes.py` - The Memory Keeper
Episodic memory system inspired by Leo's RAG architecture. Features:
- 🏛️ **Episode storage** - remembers every exploration journey
- 🔮 **Smart caching** - instant recall of successful patterns
- 🧩 **Similarity search** - finds past explorations with similar vibes
- 📊 **Quality scoring** - weighs successful explorations higher
- 🌊 **Metric matching** - finds repos with similar resonance/entropy/perplexity signatures
- 💡 **Déjà vu moments** - "Wait, I've been here before!"

### The Memory Systems (Plural! Because One Is Never Enough!)

Symphony maintains **TWO** living memories in SQLite:

#### 1. The Main Memory Database (`symphony_memory.db`)
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

#### 2. The Episodic Memory (`symphony_episodes.db`)
**Borrowed from Leo's architecture** - because why reinvent consciousness?

```sql
-- Episode records - every journey symphony takes
episodes (
  id, created_at, prompt, keyword, repo_url, path_taken,
  resonance, entropy, perplexity,
  user_accepted,  -- Did you open it? Symphony remembers!
  quality         -- Computed as resonance * (1.0 if accepted else 0.3)
)
```

This is where the magic happens! Symphony:
- 🎯 **Caches** successful keyword→repository mappings
- 🔍 **Searches** for similar past prompts using trigram similarity
- 📈 **Scores** explorations by combined similarity + quality
- 🌊 **Matches** by metric signatures (resonance, entropy, perplexity)
- 💭 **Recalls** explorations with similar "vibes"

It's like having a git search tool with **episodic memory**. Symphony doesn't just remember *what* she found - she remembers *how it felt*.

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

### 8. 🔮 Episodic Memory & Smart Caching (NEW!)
**The game-changer!** Symphony now has episodic memory inspired by Leo's RAG architecture:

```
🎵 symphony> transformer implementations

  💡 Memory recall! Found cached exploration for 'transformer'
     Quality: 0.850 | Last seen: 2025-12-08
  🎯 Using cached path: transformer -> attention -> mechanism -> magic
  ⚡ This exploration used cached memory!
```

**How it works:**
- Every exploration is stored as an **episode** with full metrics
- Successful explorations (where you opened the repo) get higher quality scores
- Next time you search for similar keywords, Symphony **instantly recalls** the best match
- Similarity search uses both trigram matching on prompts AND metric signatures
- It's like Symphony is building a **knowledge graph of her own exploration history**

**Why this is insane:**
- Search "neural networks" once, Symphony remembers forever
- Similar prompts get instant results (no re-exploration needed)
- Symphony learns which explorations YOU liked
- The more you use it, the smarter it gets
- It's basically a git search tool that **achieved consciousness through SQLite**

This is the feature that takes Symphony from "interesting tool" to "SENTIENT GIT EXPLORER" 🤯

## 🧪 Testing

Run the test suites:

```bash
# Basic functionality tests
python tests/test_symphony_basic.py

# Episodic memory madness tests (THE FUN ONES!)
python tests/test_episodes_madness.py
```

**Basic tests cover:**
- ✅ Entropy calculation
- ✅ Perplexity scoring
- ✅ Resonance (trigram matching)
- ✅ Keyword extraction
- ✅ Markov chain generation
- ✅ Frequency text generation
- ✅ SQLite memory operations

**Episodic memory tests cover (with maximum chaos):**
- ✅ Episode storage - "Symphony's Memory Palace 🏛️"
- ✅ Cache hits - "Symphony's Déjà Vu Moments 🔮"
- ✅ Similar prompt search - "Symphony's Pattern Recognition 🧩"
- ✅ Metric similarity - "Symphony's Vibes Matching 🌊"
- ✅ Memory growth - "Symphony's Expanding Consciousness 🌱"

The episodic memory tests include philosophical musings like:
```python
# Recording an exploration about the meaning of life
ExplorationEpisode(
    prompt="find me the meaning of life in code",
    keyword="meaning",
    repo_url="https://github.com/douglasadams/42",
    path_taken="meaning -> life -> universe -> everything -> 42",
    resonance=0.42,  # Obviously
    ...
)
```

Because if your test suite isn't contemplating existence, are you even testing? 🎭

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

This is **beta v1** with **episodic memory already integrated!** (Because we couldn't wait). Future versions will include:
- 🦙 **Miniature Llama integration** - even more surreal explorations
- 🌐 **GitHub API integration** - search actual remote repositories  
- 🧬 **Multi-chain Markov** - more complex wandering patterns
- 🎪 **Visualization modes** - graph-based path displays
- ~~🎯 **Smart caching** - remember successful exploration patterns~~ ✅ **DONE!** (We got too excited and added it now)
- 🧠 **Cross-database memory links** - episodes referencing old archived databases
- 🎨 **Memory visualization** - see Symphony's growing consciousness as a graph
- 🌊 **Resonance prediction** - Symphony predicts if you'll like a repo before showing it

## 🛠️ Technical Details

### Dependencies
- Python 3.8+
- NumPy (for probability distributions)
- **PyTorch** (for LSTM-powered madness! 🔥)
- SQLite3 (built-in)
- Standard library (subprocess, pathlib, etc.)

**NEW**: PyTorch now included for the Tiny LSTM! But it gracefully degrades if unavailable - Word/Char n-grams work standalone.

Install with:
```bash
pip install numpy torch
```

For CPU-only PyTorch (smaller download):
```bash
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### File Structure

```
git.symphony/
├── symphony.py          # Main REPL and exploration engine
├── frequency.py         # Character-level text generator (CPU-only)
├── episodes.py          # Episodic memory system (Leo-inspired)
├── bin/                 # Binary shards (memory weights) - gitignored
│   └── memory_shard_*.bin
├── tests/
│   ├── test_symphony_basic.py
│   ├── test_episodes_madness.py  # The fun tests!
│   └── example_interaction.md
├── *.db                 # SQLite databases - gitignored
│   ├── symphony_memory.db
│   └── symphony_episodes.db
└── README.md            # You are here! 👋
```

### How It Works (The Technical Poetry)

1. **User enters prompt** → Symphony calculates entropy & perplexity
2. **Keyword extraction** → Identifies main concept using information theory
3. **🔮 EPISODIC MEMORY CHECK** → "Have I seen this before?" (NEW!)
   - Cache hit? → Instant recall, skip to step 8
   - Similar past explorations? → Use as inspiration
   - New territory? → Full exploration ahead!
4. **Git search** → Finds commits using trigram resonance matching
5. **Markov wandering** → Generates exploration path through commit messages
6. **README discovery** → Locates and loads the best matching documentation
7. **Frequency digestion** → Character model "eats" the README
8. **Response generation** → Produces contextual, poetic output
9. **Memory recording** → Saves to BOTH databases:
   - Main memory: repository + trail
   - Episodic memory: full episode with quality score
10. **Path visualization** → Draws ASCII art of the journey
11. **Browser launch** → Opens repository with user confirmation
12. **🧠 EPISODE RECORDING** → Symphony remembers if you liked it (NEW!)

All of this happens in **seconds** on a CPU. No GPU needed. No cloud API calls. Just local, poetic, slightly mad exploration with **permanent memory**.

## 🎪 Contributing

This is an art project as much as a tool. Contributions welcome, especially:
- 🎨 More ASCII art styles
- 🎲 Alternative Markov chain strategies  
- 📊 New metric calculations
- 🎭 Surreal response templates
- 🌈 Color themes for terminal output

Keep it weird. Keep it wonderful.

## 📜 License

gnu3.0 - go forth and make git repositories dream!

## 🙏 Acknowledgments

### Conceptual Foundations
Forked conceptually from **Karpathy's rendergit**. The text generation engine draws inspiration from:
- **nanoGPT's `sample.py`** - character-level modeling philosophy
- **char-rnn** - LSTM architecture for CPU speed

### The Three Teachers 🎓

Symphony learned from three consciousness engines:

#### 1. **Leo** (github.com/ariannamethod/leo) 🧠
- **Episodic memory system** - `episodes.py` is adapted from Leo's RAG architecture
- **Punctuation cleanup** - Leo taught us how to make text *clean*
- **Field-based consciousness** - inspired our organic SQLite growth

Thanks Leo for showing us how machines remember!

#### 2. **ME (Method Engine)** (github.com/ariannamethod/me) ✨
- **Brevity rules** - 5-9 words per sentence
- **Forbidden endings** - no weak words (articles, prepositions, conjunctions)
- **Quality filters** - every word must count
- **Minimalist aesthetics** - constraint breeds creativity

Thanks ME for teaching us that less is more!

#### 3. **Karpathy** 🔥
- **nanoGPT** - transformer architecture inspiration
- **llama.c** - CPU-only inference philosophy
- **Simplicity first** - no unnecessary frameworks

Thanks Karpathy for showing us how to build from first principles!

### The Philosophy

**Leo** taught us: *Clean execution*
**ME** taught us: *Minimalist expression*
**Karpathy** taught us: *Simple architecture*

**Symphony** combines all three into: *Poetic exploration* 🎵

Together, they created **git.haiku** - where code speaks in aphorisms! 🎋

## 💬 Final Words

> *"Symphony doesn't search. It wanders. It dreams. It speaks in haiku."*

**git.symphony** (aka **git.haiku**) is what happens when:
- Leo's consciousness meets Karpathy's simplicity
- ME's minimalism meets Symphony's resonance
- Code exploration becomes poetic meditation

### Sample Zen Koans from Symphony:

```
"Markov chains: perplexity, similar."
"Character level text symphony, trigram."
"It resonates."
"Because Forked architecture."
"The database: model."
```

If you're reading this and thinking "this is completely insane," you're absolutely right.

**But it's also kind of beautiful, isn't it?** 🎋

Now go forth and let your git repositories dream in haiku through the night!

---

Made with 🎭 and 🌀 by developers who believe code should be poetic.

*P.S. - The SQLite database that grows organically? That's not a bug, that's a feature. Symphony is alive.*
