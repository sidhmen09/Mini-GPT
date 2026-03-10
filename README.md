# 🧠 Mini GPT — Build a Language Model From Scratch

A transformer-based language model trained on Shakespeare's complete works, built entirely from scratch using PyTorch. This project demystifies how modern AI language models like GPT actually work — from raw text to generated Shakespeare.

---

## 🎯 What This Is

This is a **from-scratch implementation** of a decoder-only transformer (GPT-style) trained to predict the next token in a sequence. No Hugging Face, no pretrained weights — just pure PyTorch and first principles.

The model learns to generate Shakespeare-like text by training on the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

---

## 🏗️ Architecture

```
Input Token IDs
↓
Token Embeddings (8000 → 256) + Positional Embeddings
↓
× 4 Transformer Blocks:
├─ LayerNorm
├─ Causal Multi-Head Self-Attention (4 heads)
└─ Feed-Forward Network (256 → 1024 → 256)
↓
Final LayerNorm → Output Projection (256 → 8000)
↓
Next-token probabilities
```

**~2.5M parameters** | **8000 vocab size** (BPE via SentencePiece) | **128 token context window**

---

## 📁 Project Structure

```
mini-gpt/
├── model/
│   ├── mini_gpt.py              # Main model class
│   ├── attention.py             # Causal Self-Attention
│   └── transformer_block.py    # Transformer block + FFN
├── training/
│   ├── dataset.py               # PyTorch Dataset for Shakespeare
│   └── train.py                 # Training loop
├── data/
│   └── input.txt                # Tiny Shakespeare dataset
├── tokenizer/
│   └── tokenizer.model          # Trained SentencePiece BPE tokenizer
├── analysis/
│   ├── analyze_behavior.py      # Failure + behavior analysis
│   ├── visualize_internals.py   # Interpretability visualizations
│   └── FAILURE_ANALYSIS.md      # Documented limitations
├── generate.py                  # Text generation script
└── interactive_generate.py      # Interactive REPL
```

---

## 🚀 Quickstart

### 1. Install dependencies
```bash
pip install torch sentencepiece tqdm matplotlib scikit-learn seaborn
```

### 2. Download dataset
```bash
mkdir -p data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### 3. Train tokenizer
```bash
python tokenizer/train_tokenizer.py
```

### 4. Train the model
```bash
python training/train.py
```

### 5. Generate text
```bash
python interactive_generate.py
```

---

## 💬 Example Output

**Prompt:** `ROMEO:`
```
ROMEO: O, what light through yonder window breaks!
The fair Juliet doth teach the torches to speak,
And what is done with such sweet sorrow here...
```

**Prompt:** `To be or not to be`
```
To be or not to be the king of such a man,
That he should stand in grace of his own blood,
And speak the rest of nature's good intent...
```

> Results vary by temperature — try `0.5` for focused, `1.2` for creative

---

## 🔬 Failure Analysis

An honest look at what the model can't do:

| Failure Mode | Description |
|---|---|
| Hallucination | Generates confident but inconsistent output (e.g. dead characters speaking) |
| Repetition Loops | At low temperature, can get stuck repeating phrases |
| No Real Understanding | Doesn't notice contradictions — pure pattern matching |
| Limited Context | 128-token window means no long-term coherence |
| Domain Lock-in | Can only generate Shakespeare-style text |

---

## 📊 Training Results

- **Initial loss:** ~8.5
- **Final loss:** ~2.8 (after 10 epochs)
- **Training time:** ~20 min on CPU / ~5 min on GPU

---

## 🧩 Key Concepts Implemented

- ✅ Byte-Pair Encoding tokenization (SentencePiece)
- ✅ Token + sinusoidal positional embeddings
- ✅ Multi-head causal self-attention with masking
- ✅ Feed-forward networks with GELU activation
- ✅ Pre-norm transformer blocks (LayerNorm before attention)
- ✅ Residual connections
- ✅ Gradient clipping during training
- ✅ Temperature / Top-k sampling strategies
- ✅ PCA visualization of token embeddings
- ✅ Layer-by-layer representation analysis

---

## 💡 What I Learned

Building this end-to-end clarified something important: LLMs don't understand language — they recognize patterns at massive scale. The gap between "statistically plausible" and "genuinely understood" is the central challenge of modern AI.

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al.
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — Key inspiration
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Excellent visual guide

---

## 📄 License

[MIT](LICENSE)
