# Chinese Text Complexity Scorer

**f(sentence) → difficulty_score**

A Rust-based, scientifically-grounded difficulty scorer for Chinese text using HSK/TOP vocabularies.

## The Approach

**No corpus. No generative LLM. Pure scoring.**

### 1. Extract Words
Use Aho–Corasick to match all HSK and TOP words in a sentence with their difficulty levels.

### 2. Compute Lexical Coverage  
For each learner level **L**, calculate:
```
coverage = known_words / total_words
```
(Where known_words have difficulty ≤ L)

### 3. Generate Labels via 90% Rule
Scientifically-established SLA research shows:
- **≥90% coverage** → Learner can understand (label: 1)  
- **<90% coverage** → Comprehension breaks (label: 0)

This gives you **training labels automatically** from the vocabulary alone.

### 4. Feature Engineering
Extract per-sentence features:
- Max/mean word difficulty (HSK and TOP separately)
- Count of high/low difficulty words
- Ratio of OOV words
- Sentence length
- Character rarity metrics

### 5. Train a Tiny Model
Small MLP (2 hidden layers, 32–64 units) learns:
```
F(sentence_features) → P(learner level L understands this)
```

### 6. Score at Runtime
```
Input:  "扒开橘子，把皮剥下来。"
Output: 
  Level 1: 5%   (not readable)
  Level 2: 22%  (not readable)
  Level 3: 67%  (struggling)
  Level 4: 94%  ✓ READABLE (true level)
  Level 5: 99%
  Level 6: 100%
```

## Result
**A stable, scientifically valid, real-time difficulty estimator** — no corpus required, fully deterministic.

## Usage

### Runtime Inference
```bash
# Score a sentence (requires hsk_dictionary.json, top_dictionary.json, trained model)
cargo run --bin zho-complexity --release -- "你好世界"
```

### Training Pipeline (optional - for retraining)
```bash
# 1. Prepare training dataset from sentences
cargo run --bin prepare_training_data --release

# 2. Extract BKRS difficult phrases for hard examples
cargo run --bin extract_bkrs_phrases --release

# 3. Train the model
cargo run --bin train --release
```

## Architecture

- **`src/lib.rs`** — Core types (WordMatch, SentenceFeatures, DifficultyModel)
- **`src/matcher.rs`** — Aho–Corasick word matching with HSK/TOP vocabularies
- **`src/features.rs`** — Feature extraction (12-dim vectors) + label computation (90% rule)
- **`src/model.rs`** — CORAL ordinal regression neural network
- **`src/main.rs`** — Runtime inference CLI

### Training Utilities (optional)
- **`src/bin/prepare_training_data.rs`** — Feature extraction + labeling pipeline
- **`src/bin/train.rs`** — Candle MLP training loop with Adam optimizer
- **`src/bin/extract_bkrs_phrases.rs`** — Extract difficult classical phrases

All in **Rust + Candle 0.8**. No Python. No external models.

