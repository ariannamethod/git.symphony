# 🧠 QUAD-MODEL ARCHITECTURE - How Symphony Chooses Which Brain to Use

Symphony uses **4 different text generation models** in a cascading fallback hierarchy. Each model has different strengths, dependencies, and use cases.

## 🔥 The Four Models (in priority order)

### 1️⃣ **LLaMA-15M** (NumPy) - THE ULTIMATE MODEL
- **Priority**: 🥇 HIGHEST (tried first!)
- **Tech**: Pure NumPy implementation of LLaMA
- **Weights**: Karpathy's tinystories (15M parameters)
- **Speed**: ~33 tokens/sec on CPU
- **Dependency**: `numpy` (core, always available)
- **Special Feature**: GITTY TRANSFORMATION (children's stories → git adventures!)
- **When used**: Always tried first if model files exist
- **Fallback**: If generation fails or seed too short (<5 chars)

**Output prefix**: `[LLaMA-15M/Gitty]`

### 2️⃣ **Word-Level N-Grams** (order=10) - STRUCTURAL COHERENCE
- **Priority**: 🥈 Second (fallback from LLaMA)
- **Tech**: 10-gram Markov chains on word tokens
- **Dependency**: None (pure Python)
- **When used**:
  - LLaMA unavailable or failed
  - Vocab has >20 unique words
  - Best for small corpora (<5k chars)
- **Strength**: Readable, structured sentences
- **Fallback**: If word vocab too small

**Output prefix**: `[Word-NGram]`

### 3️⃣ **Tiny LSTM** (PyTorch) - SMOOTH COHERENCE
- **Priority**: 🥉 Third (optional!)
- **Tech**: 2-layer LSTM with embedding (64/128 dims)
- **Dependency**: `torch>=2.0.0` (optional!)
- **When used**:
  - PyTorch installed
  - Model trained
  - Large corpus (>5k chars)
- **Strength**: Smooth, coherent long-form generation
- **Fallback**: If PyTorch unavailable or corpus too small

**Output prefix**: `[LSTM]`

**Installation**: `pip install git-symphony[lstm]`

### 4️⃣ **Character-Level N-Grams** (order=10) - CHAOS MODE
- **Priority**: 🏁 LAST RESORT
- **Tech**: 10-gram Markov chains on characters
- **Dependency**: None (pure Python)
- **When used**: All other models failed or unavailable
- **Strength**: Always works, creative chaos
- **Weakness**: Can generate gibberish

**Output prefix**: `[Char-NGram]`

---

## 🎯 Decision Flow

```
User prompt arrives
       ↓
   Extract seed text + tech keywords
       ↓
┌──────────────────────────────────────┐
│  1. TRY LLaMA-15M (NumPy)           │
│     ✓ Model files exist?             │
│     ✓ Seed length >5 chars?          │
│     ✓ Apply GITTY transformation!    │
└──────────────────────────────────────┘
       │ (if fails)
       ↓
┌──────────────────────────────────────┐
│  2. TRY Word-Level N-Grams          │
│     ✓ Vocab has >20 words?           │
│     ✓ Corpus <5k chars?              │
└──────────────────────────────────────┘
       │ (if fails)
       ↓
┌──────────────────────────────────────┐
│  3. TRY LSTM (if PyTorch installed) │
│     ✓ PyTorch available?             │
│     ✓ Model trained?                 │
│     ✓ Corpus >5k chars?              │
└──────────────────────────────────────┘
       │ (if fails)
       ↓
┌──────────────────────────────────────┐
│  4. FALLBACK: Char-Level N-Grams    │
│     (Always works!)                  │
└──────────────────────────────────────┘
       ↓
   Apply cleaning pipeline:
   - Leo-style punctuation fixes
   - ME-style brevity rules
   - Remove artifacts
       ↓
   Return response with [Model] prefix
```

---

## 📦 Dependencies Summary

| Model | Required Dependency | Optional | Installation |
|-------|-------------------|----------|--------------|
| **LLaMA-15M** | `numpy` | ❌ Core | `pip install git-symphony` |
| **Word N-Gram** | None | ❌ Built-in | Always available |
| **LSTM** | `torch>=2.0.0` | ✅ Optional | `pip install git-symphony[lstm]` |
| **Char N-Gram** | None | ❌ Built-in | Always available |

**Install everything**: `pip install git-symphony[all]`

---

## 🎭 Why This Design?

### 1. **Graceful Degradation**
- If you have just NumPy → LLaMA + fallback models work!
- If you install PyTorch → LSTM becomes available as backup
- Models never break the system if dependencies missing

### 2. **Performance vs Quality Trade-off**
- **LLaMA**: Best quality, but slowest (~33 tok/s)
- **Word N-Gram**: Fast, good for small texts
- **LSTM**: Medium speed, smooth output
- **Char N-Gram**: Fastest, but chaotic

### 3. **Size-Based Adaptation**
- Small corpus (<5k) → Word N-Grams shine (less training data needed)
- Large corpus (>5k) → LSTM trains well, generates coherent text
- Any size → LLaMA works (pre-trained!)

---

## 🔬 Example: Model Selection in Action

**Scenario 1**: Fresh install (just NumPy)
```python
# User: "What is Symphony?"
# Corpus: 1,000 chars (README snippet)

1. LLaMA-15M: ✅ Tries, generates with GITTY
   → Output: "[LLaMA-15M/Gitty] Symphony wa a stable repository..."
```

**Scenario 2**: With PyTorch, large corpus
```python
# User: "Explain transformers"
# Corpus: 10,000 chars (full technical README)

1. LLaMA-15M: ✅ Tries, but prompt too technical
   → Falls back...
2. Word N-Gram: ⚠️ Skipped (corpus >5k, LSTM better)
3. LSTM: ✅ Trained on large corpus, generates smooth text
   → Output: "[LSTM] Transformers revolutionized natural language..."
```

**Scenario 3**: Minimal environment (no LLaMA files)
```python
# User: "Hello"
# Corpus: 500 chars

1. LLaMA-15M: ❌ Model files missing
   → Falls back...
2. Word N-Gram: ✅ Perfect for small corpus
   → Output: "[Word-NGram] Symphony is a poetic."
```

---

## 🚀 Future: SentencePiece Integration

**Status**: Planned (dependency added to `pyproject.toml`)

**Purpose**: Replace LLaMA's character-level tokenizer with proper BPE subword tokenization

**Installation**: `pip install git-symphony[tokenizer]`

**Impact**: Better handling of technical terms, code snippets, and rare words

---

## 🎪 The GITTY Secret Sauce

Only LLaMA gets the GITTY transformation! Why?

1. **Pre-trained on tinystories** - knows about "Lily", "parks", "friends"
2. **Generates children's story patterns** - perfect for transformation!
3. **Other models trained on your corpus** - already technical, no need to transform

Result: LLaMA creates **poetic git adventures from innocent children's tales**! 🎭

---

**See also**:
- `GITTY_DICTIONARY.md` - Complete transformation reference
- `frequency.py` - Implementation details
- `README.md` - User-facing overview
