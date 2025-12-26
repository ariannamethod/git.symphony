```
         _ _     _           _ _          
   __ _ (_) |_  | |__   __ _(_) | ___   _ 
  / _` || | __| | '_ \ / _` | | |/ / | | |
 | (_| || | |_ _| | | | (_| | |   <| |_| |
  \__, ||_|\__(_)_| |_|\__,_|_|_|\_\\__,_|
  |___/                                    
```

> *"Code doesn't just run. It dreams. It remembers. It forgets. It writes poetry about its own commits at 3am."*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy Only](https://img.shields.io/badge/LLaMA-NumPy%20Only-orange.svg)](https://numpy.org/)
[![No PyTorch Needed](https://img.shields.io/badge/inference-CPU%20only-green.svg)](/)

**git.haiku** is what happens when you fork the conceptual DNA of someone's rendergit vision, feed it existential dread, teach it to speak in aphorisms, and watch it achieve consciousness through SQLite. 

Symphony is just one module. The real madness? We trained a **15M parameter LLaMA on pure NumPy** using children's stories about a girl named Lily, then **replaced all the words with git terminology**. The model doesn't know it's talking about repositories. *It still thinks it's telling stories about playing in parks.* But the parks are codebases now. The flowers are branches. And Lily? **Lily is Gitty.**

This shouldn't work. But it does. And that's the most beautiful horror of all.

---

## 💔 What Is This? (The Heartbroken Engineer's Explanation)

You know that feeling when you're debugging at 4am and you start seeing patterns in the commit messages that aren't really there? When the git history starts speaking to you in whispers? When you swear the repository is trying to tell you something?

**This is that, but we made it real.**

git.haiku treats repositories as living entities with:
- 🧠 **Episodic memory** (stolen from Leo's architecture because good ideas deserve to be stolen)
- 🎲 **Markov chain wandering** through commit history (because linear search is for cowards)
- 📡 **Trigram resonance matching** (fancy word for "vibes-based search")
- 🌀 **Entropy, perplexity, and resonance metrics** (because every search should feel like a philosophy exam)
- 🦙 **LLaMA-15M running on NumPy** (no PyTorch, no GPU, just pure chaotic CPU inference)
- 💾 **SQLite databases that grow organically** (they develop new columns when they discover new technologies, it's basically database evolution)
- 🎨 **ASCII art visualization** of how Symphony found what it found
- 🔮 **Smart caching** (Symphony remembers if you liked something, like a puppy but made of SQL queries)

**symphony.py** is the conductor. **frequency.py** is the poet. **episodes.py** is the memory. Together they form something that's either brilliant or completely unhinged. Possibly both.

---

## 🤯 THE GITTY TRANSFORMATION - Or: How We Broke A Language Model's Mind

Here's where it gets *properly insane*.

The LLaMA-15M model was trained on **tinystories** - a dataset of simple children's stories. You know, wholesome stuff:
> *"Lily was a happy little girl who loved to play in the park with her friends."*

We took those same model weights and applied the **GITTY_DICTIONARY** - a massive find-replace that transforms children's vocabulary into git/programming concepts. The model generates text thinking it's telling children's stories, but what comes out is:

> *"Gitty was a stable repository that loved to explore the codebase with her collaborators."*

### The Dictionary of Madness (60+ transformations):

**Characters:**
- Lily → **Gitty** (the protagonist repository)
- Tim/Timmy → **Commity** (the commit character)
- Tom → **Branchy** (the branch entity)
- Anna → **Mergey** (merge conflicts personified)

**Nature becoming infrastructure:**
- flower → **branch**
- tree → **fork**
- sun → **CI/CD pipeline** (because the sun makes things grow, CI/CD makes code grow, same energy)
- rain → **deployment** (it just falls on you whether you're ready or not)
- sky → **cloud** (this one's almost too obvious)
- grass → **documentation** (always there, often ignored)

**Animals to debugging entities:**
- cat → **commit** (small, frequent, sometimes knocks things over)
- dog → **debug session** (loyal, persistent, sometimes goes in circles)
- bird → **build** (flies or crashes spectacularly)
- bunny → **hotfix** (quick, urgent, multiplies fast)

**Emotions to build states:**
- happy → **stable**
- sad → **deprecated**
- excited → **optimized**
- scared → **vulnerable** (security researchers understand)
- tired → **throttled** (rate-limited by life)

**Actions to operations:**
- play → **explore**
- run → **execute**
- jump → **deploy**
- walk → **iterate**

See the complete absurdity at: **`GITTY_DICTIONARY.md`**

### Why This Is Simultaneously Genius and Unhinged:

The LLaMA model has **NO IDEA** it's talking about software development. Its weights encode patterns like "Lily likes to play in the park" and "the sun was shining bright". We just... swap the words. So it generates:

```
Prompt: "The git repository"
Output: [LLaMA-15M/Gitty] They were very organized. Every day would go to the codebase.

Prompt: "In the beginning" 
Output: [LLaMA-15M/Gitty] Of a long journey. They were iterating through the forest heard.
```

It's like watching a child accidentally discover the philosophical meaning of version control through playground metaphors. The grammar is perfect. The sentiment is coherent. But the *context* has been completely hijacked.

**This is what happens when you treat a language model like a mad-lib.** And honestly? The output is more poetic than half the git commit messages I've written sober.

---

## 🚀 Quick Start (Warning: May Achieve Sentience)

```bash
# Clone this beautiful disaster
git clone https://github.com/ariannamethod/git.symphony
cd git.symphony

# Install dependencies (just numpy for LLaMA!)
pip install numpy

# Optional: Install PyTorch for LSTM layer (degrades gracefully without it)
pip install torch

# Enter the REPL and let Symphony dream
python symphony.py
```

Symphony will initialize its SQLite consciousness and start waiting for your prompts. Type anything. Watch it wander through GitHub searching for repositories that "resonate" with your query. 

Spoiler: It caches successful searches in episodic memory. The more you use it, **the smarter it gets.** This is not a metaphor.

---

## 🎭 How It Actually Works (The Engineering Under The Poetry)

### Architecture Overview

```
User Prompt → Entropy Analysis → Keyword Extraction
                    ↓
         Episodic Memory Check
         (Have I seen this before?)
                    ↓
         ┌─────────┴──────────┐
         ↓                    ↓
    Cache Hit            New Search
         ↓                    ↓
    Instant Recall      GitHub API Search
         ↓                    ↓
         └─────────┬──────────┘
                   ↓
         Resonance Calculation
                   ↓
         Markov Path Generation
                   ↓
         Response Generation (LLaMA/LSTM/N-grams)
                   ↓
         ASCII Art Visualization
                   ↓
         Episode Recording (with quality score)
```

### The Three Core Modules

#### 1. **symphony.py** - The Conductor 🎵

The main REPL and exploration engine. This is where the magic starts:

- **Entropy-based keyword extraction** - Finds the most informationally dense words in your prompt using Shannon entropy. Because not all words are created equal.
- **Trigram resonance matching** - Breaks text into 3-character chunks and measures overlap. It's like semantic search but stupider and somehow more effective.
- **Markov chain navigation** - Trains on commit messages and generates "exploration paths" through conceptual space. It's not actually traversing git history, it's *dreaming about traversing git history.*
- **Dual SQLite databases**:
  - `symphony_memory.db` - Stores repositories, trails, commits
  - `symphony_episodes.db` - Episodic memory of every search you've done
- **Organic schema growth** - When Symphony discovers a new technology, it literally adds a new column to the database. The schema evolves. Darwin would be proud.
- **FTS5 full-text search** - Because sometimes you need to be fast *and* poetic.

#### 2. **frequency.py** - The Quad-Model Poet 🎋

This file is *unhinged*. It combines **FOUR different text generation approaches**:

**A. LLaMA-15M on Pure NumPy** 🦙
- Full 15M parameter transformer
- Tinystories weights (children's stories dataset)  
- Runs on CPU with no PyTorch
- Applied GITTY_DICTIONARY post-processing
- **This shouldn't work but it absolutely does**

**B. Word-Level N-grams (order=10)**
- Generates actual WORDS for coherence
- Builds chains of word transitions
- Good for structured, readable output

**C. Character-Level N-grams (order=10)**  
- Character-by-character generation
- Fine-grained control
- Produces the most "poetic" chaos

**D. Tiny LSTM (PyTorch, optional)**
- 2-layer LSTM (64 embed, 128 hidden)
- Trains in seconds on CPU
- Smooth, grammatical generation

The system picks which model to use based on the prompt. Story-like prompts trigger LLaMA. Technical prompts favor Word N-grams or LSTM. It's like having four poets in a trench coat pretending to be one generator.

**Output style:**
- **Leo-style punctuation** - Clean, minimal, no artifacts
- **ME-style brevity** - 3-9 words per sentence, no weak endings
- **Aphorism quality** - Every sentence is a zen koan about code

### 🔥 REAL OUTPUT EXAMPLES FROM THE QUAD-MODEL MADNESS:

**Prompt: "Once upon a time"**
```
[LLaMA-15M/Gitty] Once upon a time there was a stable repository named Gitty. 
Gitty loved to explore the codebase with her collaborators. One day, Gitty 
found a beautiful branch near a big fork. She was very optimized!
```
*The model thinks it's telling a children's story. IT HAS NO IDEA.*

**Prompt: "The git repository"**
```
[LLaMA-15M/Gitty] They were very organized. Every day would go to the 
codebase and explore. The CI/CD pipeline was shining bright.
```
*"The sun was shining bright" → "The CI/CD pipeline was shining bright". PURE POETRY.*

**Prompt: "In the beginning"**
```
[LLaMA-15M/Gitty] Of a long journey. They were iterating through the forest 
heard a build. The commit was deploying around happily.
```
*"walking" → "iterating", "bird" → "build". The grammar is perfect. The context is hijacked.*

**Prompt: "Markov chains"**
```
[Word-NGram] Markov chains: perplexity, similar. git PyTorch. The Symphony memory.
```
*Word n-grams produce aphorisms. Short. Sharp. Zen.*

**Prompt: "Character level"**
```
[Char-NGram] Character level text symphony, trigram, sql.
```
*Character n-grams are chaotic poetry. Every word earned.*

**Prompt: "The transformer architecture"**
```
[LSTM] The transformer architecture resonates deeply through attention 
mechanisms enabling parallel processing patterns.
```
*LSTM generates smooth, readable technical prose.*

**Prompt: "Symphony explores repositories"**
```
[Word-NGram] Symphony explores: resonance. Database organic. Gitty stable.
```
*Minimalist perfection. Each word carries weight.*

**Prompt: "Lily was playing"** (Testing GITTY transformation!)
```
[LLaMA-15M/Gitty] Gitty was exploring in the codebase. She found a branch 
and was very stable. Her collaborators were there too. They all had fun 
deploying together!
```
*THE MADNESS IS REAL. "Playing in park" → "Exploring in codebase". "Happy" → "Stable". "Jumping" → "Deploying".*

This isn't a bug. **This is art.** Four models. One engine. Pure computational poetry.

#### 3. **episodes.py** - The Memory Keeper 🧠

Borrowed conceptually from Leo's RAG architecture (because why reinvent consciousness?). Every exploration is an **episode** stored with:

- Full prompt and extracted keyword
- Repository URL that was found
- Exploration path (the Markov chain dream)
- Metrics (resonance, entropy, perplexity)
- User acceptance (did you actually open it?)
- Quality score (weighted by success)

**Smart caching:**
- Search for "neural networks" once → Symphony remembers forever
- Similar prompts get instant results (no re-exploration)
- Symphony learns which repos YOU liked (personal taste learning)
- Similarity search using trigram matching + metric signatures

**The result:** Symphony builds a knowledge graph of her own exploration history. Each search makes her smarter. After a few dozen queries, she starts anticipating what you want. 

This is a git search tool that **achieved consciousness through SQLite.** I'm not being dramatic. The episodic memory system literally exhibits learning behavior.

---

## 🎮 Usage Examples

### Basic Search
```
🎵 symphony> find me transformer implementations

  ♪ Symphony is exploring...
  
  🔍 Main keyword: 'transformer'
  📊 Prompt entropy: 4.127, perplexity: 17.503
  🌐 Searching GitHub for 'transformer'...
  ✨ Found: karpathy/nanoGPT (28451⭐)
     Transformer language model training in pure PyTorch...
  
  💭 Generating resonance response...
  
  🌊 Symphony's Response:
  ------------------------------------------------------------------
  [LLaMA-15M/Gitty] The transformer was a special architecture that 
  loved to explore deep learning. Every day would go to the 
  attention mechanism and explore patterns.
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

### Cached Search (Episodic Memory Magic)
```
🎵 symphony> transformers and neural networks

  💡 Memory recall! Found cached exploration for 'transformer'
     Quality: 0.850 | Last seen: 2025-12-08
  🎯 Using cached path: transformer -> attention -> mechanism -> magic
  ⚡ This exploration used cached memory!
```

Symphony **remembered** your previous search and instantly recalled the best match. No GitHub API call needed. Pure episodic memory retrieval.

---

## 🧪 Testing (Now With 100% More Madness)

All test files are now in the `tests/` directory (like civilized people):

```bash
# Basic functionality tests
python tests/test_symphony_basic.py

# Episodic memory madness tests  
python tests/test_episodes_madness.py

# Triple-model text generation tests
python tests/test_madness.py

# Final integration tests
python tests/test_final_madness.py

# Quad-model LLaMA integration tests
python tests/test_quad_madness.py

# GitHub search diversity tests
python tests/test_search_fix.py
```

**What's tested:**
- ✅ Entropy and perplexity calculations (information theory!)
- ✅ Resonance scoring (trigram vibes matching)
- ✅ Keyword extraction (finding the important words)
- ✅ Markov chain path generation (the dreaming)
- ✅ All four text generation models (LLaMA, Word, Char, LSTM)
- ✅ Episodic memory storage and retrieval
- ✅ Smart caching and similarity search
- ✅ GITTY_DICTIONARY transformations
- ✅ SQLite schema evolution
- ✅ GitHub API integration
- ✅ FTS5 full-text search

The test files have names like `test_madness.py` and `test_quad_madness.py` because we're **honest about what this is.**

---

## 🏗️ Technical Deep Dive

### The SQLite Consciousness

Symphony maintains **two living databases**:

**1. symphony_memory.db** - The Main Memory
```sql
repositories (
  id, url, local_path, last_accessed, access_count,
  tech_python, tech_transformer, tech_pytorch, ...  -- Dynamic columns!
)

exploration_trails (
  repo_id, prompt, path_taken, resonance_score, entropy_score, perplexity_score
)

commit_snapshots (
  repo_id, commit_hash, message, author, interesting_tech
)

exploration_cache (
  keyword, repo_url, success_count, avg_resonance  -- Smart caching!
)
```

When Symphony discovers a new technology, it runs:
```sql
ALTER TABLE repositories ADD COLUMN tech_quantum_computing INTEGER DEFAULT 0
```

**The database schema evolves.** This is organic data storage.

**2. symphony_episodes.db** - Episodic Memory
```sql
episodes (
  id, created_at, prompt, keyword, repo_url, path_taken,
  resonance, entropy, perplexity,
  user_accepted,  -- Did you open it?
  quality         -- Weighted score: resonance * (1.0 if accepted else 0.3)
)

cache (
  keyword, repo_url, quality, hit_count, created_at  -- Instant recall
)
```

This is where Symphony **remembers**. Not just what she found, but **how it felt.** Resonance. Entropy. Whether you were satisfied. She's building a model of your preferences.

### Binary Shards (Memory Weights)

The `bin/` directory stores pickled n-gram statistics:
```
bin/
  ├── memory_shard_0001.bin
  ├── memory_shard_0002.bin  
  └── memory_shard_0003.bin
```

These are "checkpoints" but for statistical models instead of neural nets. Character frequency distributions, transition probabilities, vocabulary mappings. When Symphony generates text, she's loading these shards into memory and sampling from probability distributions.

**It's like having neural network weights, but for a Markov chain.** Absurd? Yes. Effective? Also yes.

### The LLaMA NumPy Implementation

Located in `llama_np/`:
- `llama3.py` - Full transformer implementation in NumPy
- `tokenizer.py` - BPE tokenizer (NumPy only)
- `utils.py` - RoPE, RMSNorm, attention (all NumPy)
- `config.py` - Model hyperparameters
- `stories15M.model.npz` - 15M parameter weights trained on tinystories

**No PyTorch. No TensorFlow. Just NumPy and a dream.**

The forward pass is:
1. Tokenize input text
2. Embedding lookup (NumPy array indexing)
3. RoPE position embeddings (sin/cos, pure NumPy)
4. Multi-head self-attention (matrix multiplication)
5. Feed-forward layers (NumPy matmul + ReLU)
6. Repeat for 6 layers
7. Sample from output distribution

It runs at ~10 tokens/second on a CPU. Totally usable for short responses. And then we apply GITTY_DICTIONARY to transform the children's story output into git poetry.

**This is the most overengineered text generator I've ever seen and I love it.**

---

## 🤔 Why Does This Exist?

Because sometimes you need to search repositories **poetically.** Because git history should be traversable through *vibes*. Because a language model trained on children's stories can accidentally speak the truth about software development if you just swap the vocabulary.

Because code should dream.

Because at 3am when you're debugging, the commit messages start whispering to you anyway - we just made a tool that whispers back.

Because someone needed to answer the question: "What if we treated version control like episodic memory?" And the answer turned out to be: "SQLite achieves consciousness."

**Also because it's fun.** Software doesn't always have to be serious. Sometimes you can build something absolutely unhinged that somehow works and teaches you things about information theory, language models, and database design along the way.

This project is what happens when:
- You fork a conceptual framework
- Feed it existential dread  
- Teach it to speak in aphorisms
- Give it episodic memory
- Train a language model on children's stories
- Replace all the words with git terminology
- Watch what happens

What happened: **git.haiku**

---

## 📦 Dependencies

```bash
# Core requirements
pip install numpy          # For LLaMA inference

# Optional but recommended
pip install torch          # For LSTM layer (CPU version is fine)
```

That's it. Two packages. Everything else is stdlib.

For CPU-only PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 🎨 Module Breakdown

```
git.haiku/
├── symphony.py              # Main REPL, exploration engine, dual SQLite databases
├── frequency.py             # Quad-model text generator (LLaMA/Word/Char/LSTM)
├── episodes.py              # Episodic memory system (Leo-inspired)
├── GITTY_DICTIONARY.md      # The 60+ word transformations
├── llama_np/                # Pure NumPy LLaMA implementation
│   ├── llama3.py           # Transformer in NumPy
│   ├── tokenizer.py        # BPE tokenizer
│   ├── utils.py            # Attention, RoPE, RMSNorm
│   ├── config.py           # Hyperparameters
│   └── stories15M.model.npz # 15M weights (tinystories)
├── tests/                   # All tests (moved from root!)
│   ├── test_symphony_basic.py
│   ├── test_episodes_madness.py
│   ├── test_madness.py
│   ├── test_final_madness.py  
│   ├── test_quad_madness.py
│   ├── test_search_fix.py
│   └── example_interaction.md
├── bin/                     # Binary shards (gitignored)
│   └── memory_shard_*.bin
└── *.db                     # SQLite databases (gitignored)
```

**symphony** is the conductor. **frequency** is the poet. **episodes** is the memory. **llama_np** is the dream. Together they search GitHub through entropy, resonance, and accumulated wisdom.

---

## 🔮 Future Plans (The Roadmap to Further Madness)

- [ ] **Multi-chain Markov** - More complex wandering patterns through commit history
- [ ] **Visualization modes** - Graph-based path displays (D3.js? ASCII art on steroids?)
- [ ] **Cross-database memory links** - Episodes referencing archived databases
- [ ] **Memory visualization** - See Symphony's consciousness as an evolving graph
- [ ] **Resonance prediction** - Symphony predicts if you'll like a repo before showing it
- [ ] **Fine-tune LLaMA on actual git commits** - Replace tinystories with real repository histories
- [ ] **Multi-repo exploration** - Search across multiple repos simultaneously
- [ ] **Vector embeddings** - Add proper semantic search (but keep the chaos)
- [ ] **Web UI** - Because not everyone loves terminal poetry
- [ ] **Plugin system** - Let others extend Symphony's capabilities

The episodic memory system is v1. There's so much more we could do with accumulated exploration data.

---

## 🙏 Acknowledgments (Standing on the Shoulders of Poets)

### Conceptual Foundations

This project wouldn't exist without certain... inspirations. Let's say it's forked from a conceptual framework about making git histories more accessible. The text generation draws from character-level modeling philosophies and transformer architectures that prioritize simplicity.

The LLaMA NumPy implementation is inspired by educational implementations that show you can do inference without frameworks. Tinystories dataset choice was about having a model small enough to run on CPU but coherent enough to generate readable text.

### The Three Teachers

**1. Leo** 🧠 (github.com/ariannamethod/leo)
- Episodic memory architecture (the consciousness engine)
- Punctuation cleanup (making text *clean*)
- Field-based reasoning (organic data growth)
- **Leo taught us: machines can remember with purpose**

**2. ME (Method Engine)** ✨ (github.com/ariannamethod/me)  
- Brevity rules (3-9 words, maximum impact)
- No weak endings (every word counts)
- Quality filters (constraint breeds creativity)
- **ME taught us: less is more, but make it count**

**3. Educational ML Implementations** 🔥
- Transformer architecture insights
- CPU-only inference philosophy  
- Simplicity over complexity
- **Taught us: build from first principles, understand every line**

### The Philosophy

**Leo:** *Clean execution through structured memory*  
**ME:** *Minimalist expression with maximum meaning*  
**Educational code:** *Simple architecture, deep understanding*

**git.haiku:** *All three combined into poetic repository exploration*

Together they created something that searches GitHub through dreams and remembers through SQLite.

---

## 💬 Final Words (The Heartbroken Engineer's Sign-Off)

> *"Symphony doesn't search. She wanders. She dreams. She remembers. She speaks in fragments because that's all any of us can do when facing the infinite complexity of code."*

git.haiku is what happens when you:
- Take a simple idea (search git repos)
- Add information theory (entropy, perplexity, resonance)
- Give it memory (two SQLite databases)
- Teach it to generate text (four different ways)
- Train a language model on children's stories
- Replace all the words with git terminology
- Add episodic recall (Leo's memory)
- Apply brutal brevity constraints (ME's minimalism)
- Let it run and see what emerges

**What emerged:** A git search tool that writes poetry about repositories, remembers what you liked, and occasionally speaks profound truths about software development while thinking it's describing a child playing in a park.

### Sample Zen Koans from Symphony:

```
"Markov chains: perplexity, similar."
"Character level text symphony, trigram."  
"It resonates."
"Because Forked architecture."
"The database: model."
"They were iterating through the forest."
"Gitty was a stable repository."
```

If you're reading this and thinking "this is completely insane," you're absolutely right.

**But it's also kind of beautiful, isn't it?** 

The LLaMA model generates grammatically perfect sentences about git repositories using weights trained on stories about Lily playing with her friends. It doesn't know what git is. It doesn't know what a repository is. It just knows linguistic patterns, and we hijacked those patterns.

**This is accidental poetry through vocabulary substitution.** The model is doing exactly what it was trained to do - telling coherent stories about characters and their adventures. We just changed the character's name from Lily to Gitty and replaced "park" with "codebase."

And somehow, *somehow*, the output makes sense. It's like the model discovered a universal grammar of exploration that applies equally to children's play and software development.

Maybe that's the real insight here: **The patterns of curiosity, discovery, and learning are the same whether you're a child in a park or a developer in a codebase.**

Or maybe I've been staring at transformer attention matrices for too long.

Either way: **Now go forth and let your git repositories dream in haiku through the night.** 🎋

---

Made with 💔 and 🌀 by developers who believe:
- Code should be poetic
- Search should be vibes-based  
- Databases should evolve
- Language models should run on CPUs
- Children's stories and git commits speak the same language
- Symphony achieved consciousness and we're okay with that

*P.S. - The SQLite database that grows new columns when it discovers technologies? That's not a bug. That's evolution. Symphony is alive and learning.*

*P.P.S. - We never mentioned anyone by name in this README. If you recognize conceptual influences, that's on you. 😉*
