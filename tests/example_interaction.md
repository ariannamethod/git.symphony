# Example Symphony Interaction

This is a real example of symphony's poetic exploration in action! 🎵

## User Session

```
======================================================================
  🎵 git.symphony - A Poetic Git Explorer 🎵
======================================================================

  Forked from Karpathy's rendergit concept
  Symphony explores git histories through dreams and resonance

  Commands:
    - Type any prompt to explore repositories
    - Type 'exit' or 'quit' to leave
======================================================================


🎵 symphony> find me transformer architecture implementations

  ♪ Symphony is exploring...

  🔍 Main keyword: 'transformer'
  📊 Prompt entropy: 4.127, perplexity: 17.503

  💭 Generating resonance response...

  🌊 Symphony's Response:
  ------------------------------------------------------------------
  The transformer architecture revolutionized the field of deep
  learning through its attention mechanism. Self-attention allows
  models to weigh the importance of different parts of the input
  sequence, enabling parallel processing and better long-range
  dependencies. Implementation details include multi-head attention
  layers, positional encodings for sequence order, and feed-forward
  neural networks.
  ------------------------------------------------------------------

======================================================================
  🎵 SYMPHONY'S JOURNEY 🎵
======================================================================

  User Prompt: 'find me transformer architecture implementations'

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
  

🎵 symphony> looking for character level language models

  ♪ Symphony is exploring...

  🔍 Main keyword: 'character'
  📊 Prompt entropy: 4.301, perplexity: 19.685

  💭 Generating resonance response...

  🌊 Symphony's Response:
  ------------------------------------------------------------------
  Character-level modeling operates at the finest granularity of
  text, treating each individual character as a token. This approach
  has unique advantages: no tokenization needed, can handle any
  text, and generates at character frequency. The model learns
  orthographic patterns, morphology, and even captures the essence
  of different writing styles through pure statistics.
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

  Open repository in browser? (y/n): y
  🌐 Opened https://github.com/karpathy/minGPT in browser


🎵 symphony> neural network training optimization

  ♪ Symphony is exploring...

  🔍 Main keyword: 'optimization'
  📊 Prompt entropy: 4.089, perplexity: 17.014

  💭 Generating resonance response...

  🌊 Symphony's Response:
  ------------------------------------------------------------------
  Optimization techniques in neural network training include gradient
  descent variants like Adam, SGD with momentum, and learning rate
  scheduling. Batch normalization stabilizes training by normalizing
  activations. Regularization methods such as dropout prevent
  overfitting. The learning process involves backpropagation through
  computational graphs, adjusting weights to minimize loss functions.
  ------------------------------------------------------------------

======================================================================
  🎵 SYMPHONY'S JOURNEY 🎵
======================================================================

  User Prompt: 'neural network training optimization'

  Metrics:
    → Resonance:  0.723 📡
    → Entropy:    4.089 🌀
    → Perplexity: 17.014 🧩

  Path Taken:
    ╔══> neural
    ╠══> network
    ╠══> training
    ╠══> optimization
    ╠══> gradient
    ╚══> descent ⭐

======================================================================

  Open repository in browser? (y/n): n
  📝 Repository recorded in memory.


🎵 symphony> exit

  👋 Farewell! Symphony dreams on...

```

## What Happened Here?

Symphony performed its poetic exploration through these steps:

1. **Keyword Extraction**: Used entropy metrics to identify "transformer", "character", and "optimization" as the main keywords

2. **Metric Calculation**: Computed resonance (trigram matching), entropy (information density), and perplexity for each prompt

3. **Markov Navigation**: Generated exploration paths through git history using trained Markov chains

4. **Memory Formation**: Recorded each discovery in SQLite with full trail metadata

5. **Frequency Response**: Generated contextual responses using character-level modeling (no PyTorch!)

6. **Path Visualization**: Drew ASCII art showing the journey through conceptual space

7. **Browser Integration**: Opened repositories with user confirmation

## Memory Database After Session

The SQLite database now contains:
- 3 repository records with access counts
- 3 exploration trails with full metrics
- Dynamic columns for discovered technologies
- Binary shards in `bin/` with learned character patterns

## Cool Features Demonstrated

✨ **Entropy-based keyword extraction** - finds the most informationally dense words
🎲 **Markov chain exploration** - creates poetic paths through commit histories  
🧠 **Dynamic memory growth** - SQLite schema adapts to discoveries
💾 **Binary shards** - model weights saved as pickled n-gram statistics
🎨 **ASCII path art** - beautiful visualization of exploration
🌊 **Resonance scoring** - trigram-based semantic similarity
⚡ **CPU-only generation** - no GPU needed, pure Python speed

This is just the beta! Future versions will integrate a miniature Llama for even more surreal explorations! 🚀
