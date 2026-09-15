# Transformer-Based Machine Translation

A sequence-to-sequence **Transformer-based Neural Machine Translation (NMT)** system implemented using deep learning techniques. The project explores how the Transformer architecture can learn to translate sentences from a source language to a target language using self-attention and cross-attention mechanisms.

**Project:** Transformer-Based Machine Translation

---

## 🔗 W&B Report

Detailed experiment tracking, training curves, model analysis, and visualizations are available in the Weights & Biases report:

[W&B Report — Transformer-Based Machine Translation](https://wandb.ai/ee23b052-iitm-india/DA_DL_Ass2/reportlist?utm_source=chatgpt.com)

---

# 📌 Project Overview

Machine Translation is the task of automatically converting text from one natural language into another.

This project implements a **Transformer-based encoder-decoder architecture** for Neural Machine Translation.

Unlike traditional recurrent architectures such as RNNs, LSTMs, and GRUs, the Transformer relies primarily on **self-attention mechanisms**, allowing it to process tokens in parallel and model long-range dependencies more effectively.

The overall translation pipeline is:

```text
Source Sentence
       │
       ▼
Tokenization
       │
       ▼
Source Embeddings
       │
       ▼
Positional Encoding
       │
       ▼
Transformer Encoder
       │
       │  Encoder Representations
       ▼
Transformer Decoder
       │
       ▼
Linear Projection
       │
       ▼
Softmax
       │
       ▼
Target Sentence
```

---

# ✨ Key Features

* Transformer encoder-decoder architecture.
* Self-attention based sequence modelling.
* Multi-head attention.
* Positional encoding.
* Encoder-decoder cross-attention.
* Teacher forcing during training.
* Padding and look-ahead attention masks.
* Autoregressive decoding during inference.
* Cross-entropy based sequence prediction.
* Experiment tracking using Weights & Biases.
* Translation evaluation using standard machine-translation metrics.
* End-to-end training and inference pipeline.

---

# 🏗️ Project Structure

```text
transformer-machine-translation/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── positional_encoding.py
│   └── transformer.py
│
├── dataset/
│   ├── __init__.py
│   ├── tokenizer.py
│   └── dataset.py
│
├── train.py
├── evaluate.py
├── inference.py
├── utils.py
├── requirements.txt
├── README.md
└── notebooks/
    └── experiments.ipynb
```

> Update the structure above if the actual repository contains different filenames or folders.

---

# 🧠 Transformer Architecture

The Transformer follows an encoder-decoder architecture.

```text
                    Transformer
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
        Encoder Stack          Decoder Stack
             │                       │
             │                       │
      Source Sequence          Target Sequence
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                  Linear + Softmax
                         │
                         ▼
                  Target Tokens
```

---

# 🔹 1. Token Embeddings

Input sentences are first converted into tokens.

Each token is mapped to a dense vector representation:

```text
Token → Integer ID → Embedding Vector
```

For example:

```text
"I love learning"

        ↓

["I", "love", "learning"]

        ↓

[12, 48, 193]
```

The embedding layer converts these token IDs into continuous vector representations.

---

# 🔹 2. Positional Encoding

Transformers do not inherently understand the order of tokens.

Therefore, positional information is added to token embeddings.

The positional encoding allows the model to distinguish between:

```text
"dog bites man"
```

and:

```text
"man bites dog"
```

A common sinusoidal formulation is:

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))

PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

where:

* `pos` = token position
* `i` = embedding dimension index
* `d_model` = model embedding dimension

---

# 🔹 3. Self-Attention

Self-attention allows every token to interact with other tokens in the same sequence.

The attention mechanism uses three matrices:

```text
Query (Q)
Key   (K)
Value (V)
```

The attention operation is:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √dₖ)V
```

This allows the model to assign different importance to different tokens.

For example:

```text
"The cat that was sitting on the table moved."

```

The model can learn relationships between:

```text
"cat"
```

and:

```text
"moved"
```

even though they are separated by several words.

---

# 🔹 4. Multi-Head Attention

Instead of using a single attention mechanism, the Transformer uses multiple attention heads.

```text
Input
  │
  ├──► Attention Head 1
  ├──► Attention Head 2
  ├──► Attention Head 3
  ├──► ...
  └──► Attention Head N
          │
          ▼
       Concatenate
          │
          ▼
      Linear Layer
```

Different heads can learn different relationships within the sentence.

The outputs of all attention heads are concatenated and projected back to the model dimension.

---

# 🔹 5. Transformer Encoder

Each encoder block contains:

```text
Input
  │
  ▼
Multi-Head Self-Attention
  │
  ▼
Add & Norm
  │
  ▼
Feed Forward Network
  │
  ▼
Add & Norm
  │
  ▼
Output
```

The encoder processes the complete source sentence and produces contextual representations.

---

# 🔹 6. Transformer Decoder

The decoder generates the translated sentence one token at a time.

Each decoder block contains:

```text
Target Tokens
      │
      ▼
Masked Self-Attention
      │
      ▼
Add & Norm
      │
      ▼
Cross-Attention
      ▲
      │
Encoder Output
      │
      ▼
Add & Norm
      │
      ▼
Feed Forward Network
      │
      ▼
Add & Norm
```

The decoder therefore uses both:

1. Previously generated target tokens.
2. Representations produced by the encoder.

---

# 🔐 Attention Masks

Two important masks are used during Transformer training.

## Padding Mask

Padding tokens are ignored during attention computation.

For example:

```text
"I love cats <PAD> <PAD>"
```

The model should not treat `<PAD>` as meaningful information.

---

## Look-Ahead / Causal Mask

During decoding, the model must not see future target tokens.

For:

```text
I love machine learning
```

when predicting:

```text
machine
```

the decoder should only have access to:

```text
I love
```

and not:

```text
learning
```

This prevents information leakage during training.

---

# 🎯 Training

During training, the decoder receives the ground-truth target sequence shifted by one position.

For example:

```text
Target:

<BOS> I love cats <EOS>
```

Decoder input:

```text
<BOS> I love cats
```

Expected output:

```text
I love cats <EOS>
```

This is commonly known as **teacher forcing**.

---

# 📉 Loss Function

The model predicts a probability distribution over the target vocabulary for every output position.

The primary objective is token-level Cross-Entropy Loss:

```text
L = - Σ yₜ log(pₜ)
```

where:

* `yₜ` is the correct target token.
* `pₜ` is the predicted probability of the correct token.

Padding tokens are excluded from the loss calculation using the padding mask.

---

# 🔄 Training Pipeline

```text
Parallel Sentence Pairs
          │
          ▼
      Tokenization
          │
          ▼
 Add Special Tokens
          │
          ▼
       Padding
          │
          ▼
     Embeddings
          │
          ▼
 Positional Encoding
          │
          ▼
 Transformer Encoder
          │
          ▼
 Transformer Decoder
          │
          ▼
 Vocabulary Projection
          │
          ▼
 Cross-Entropy Loss
          │
          ▼
 Backpropagation
          │
          ▼
      Optimizer
          │
          ▼
    Updated Model
```

---

# 🚀 Inference

During inference, the model generates the translation autoregressively.

The process starts with the beginning-of-sentence token:

```text
<BOS>
```

The model predicts the next token.

That token is then appended to the decoder input and the process continues.

```text
<BOS>
  │
  ▼
Predict token 1
  │
  ▼
<BOS> token1
  │
  ▼
Predict token 2
  │
  ▼
<BOS> token1 token2
  │
  ▼
...
  │
  ▼
<EOS>
```

The generated sequence is finally converted back into natural language.

---

# 📊 Evaluation

Machine Translation performance can be evaluated using metrics such as:

### BLEU Score

BLEU evaluates the similarity between generated translations and reference translations using n-gram precision.

Higher BLEU generally indicates better translation quality.

### Validation Loss

Validation loss is monitored to evaluate how well the model generalizes to unseen sentence pairs.

### Qualitative Evaluation

Example translations can also be inspected manually:

```text
Source:
<source sentence>

Reference:
<ground-truth translation>

Prediction:
<model-generated translation>
```

---

# 📈 Experiment Tracking

All major experiments are tracked using **Weights & Biases (W&B)**.

The report contains experiment results, training curves, model comparisons, and visual analysis.

[Open W&B Report](https://wandb.ai/ee23b052-iitm-india/DA_DL_Ass2/reportlist?utm_source=chatgpt.com)

Typical tracked quantities include:

* Training loss
* Validation loss
* Learning curves
* Evaluation metrics
* Hyperparameter configurations
* Model performance
* Experiment comparisons

---

# ⚙️ Configuration

The major Transformer hyperparameters can be configured according to the experiment:

| Parameter            | Description                     |
| -------------------- | ------------------------------- |
| `d_model`            | Transformer embedding dimension |
| `num_heads`          | Number of attention heads       |
| `num_encoder_layers` | Number of encoder blocks        |
| `num_decoder_layers` | Number of decoder blocks        |
| `d_ff`               | Feed-forward hidden dimension   |
| `dropout`            | Dropout probability             |
| `batch_size`         | Training batch size             |
| `learning_rate`      | Optimizer learning rate         |
| `num_epochs`         | Number of training epochs       |
| `max_seq_len`        | Maximum sequence length         |

Use the values from the final experiment configuration when reproducing the reported results.

---

# 🛠️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd transformer-machine-translation
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If a requirements file is not available:

```bash
pip install torch torchvision numpy pandas matplotlib tqdm
```

Install any additional tokenizer or NLP libraries used by the implementation.

---

# ▶️ Training

Run the training script:

```bash
python train.py
```

The training process should:

1. Load the dataset.
2. Tokenize the source and target sentences.
3. Construct batches.
4. Create attention masks.
5. Run the Transformer.
6. Calculate Cross-Entropy Loss.
7. Perform backpropagation.
8. Update model parameters.
9. Evaluate on the validation set.
10. Log metrics to W&B.

---

# 🔍 Evaluation

Run:

```bash
python evaluate.py
```

The evaluation script generates predictions on the test/validation set and calculates the selected translation metrics.

---

# 💬 Inference

Run:

```bash
python inference.py
```

A sample inference workflow:

```python
source_sentence = "your source sentence"

translation = model.translate(source_sentence)

print("Source:", source_sentence)
print("Translation:", translation)
```

Adapt the function name to the actual inference implementation in the repository.

---

# 🧪 Example

```text
Input:
<source language sentence>

Expected:
<reference translation>

Model:
<predicted translation>
```

The trained model generates the target sequence autoregressively until the `<EOS>` token is produced.

---

# 📌 Why Transformers?

Traditional recurrent models process a sequence sequentially:

```text
x₁ → x₂ → x₃ → x₄ → ...
```

This makes it difficult to efficiently process long sequences and limits parallelization.

Transformers process sequence representations using attention:

```text
x₁ ─┐
x₂ ─┤
x₃ ─┼──► Self-Attention
x₄ ─┤
x₅ ─┘
```

This provides:

* Better parallelization during training.
* Direct interactions between distant tokens.
* Efficient modelling of long-range dependencies.
* A scalable architecture for sequence modelling.

---

# 🔬 Key Learning Outcomes

Through this project, the following concepts are explored:

* Neural Machine Translation
* Sequence-to-sequence learning
* Transformer architecture
* Self-attention
* Multi-head attention
* Cross-attention
* Positional encoding
* Encoder-decoder architecture
* Teacher forcing
* Attention masking
* Autoregressive decoding
* Cross-Entropy loss
* Translation evaluation
* Experiment tracking with W&B

---

# ⚠️ Limitations

Potential limitations of the system include:

* Translation quality depends strongly on the size and quality of the training corpus.
* Rare words may be difficult to translate accurately.
* Autoregressive decoding can be computationally expensive.
* Long sequences increase attention computation.
* A model trained on a limited domain may not generalize well to other domains.
* Greedy decoding may produce less optimal translations compared with beam search or other decoding strategies.

---

# 🚀 Future Improvements

Possible improvements include:

* Larger and more diverse parallel datasets.
* Subword tokenization such as BPE or SentencePiece.
* Beam-search decoding.
* Label smoothing.
* Learning-rate warm-up and scheduling.
* Larger Transformer models.
* Better regularization.
* Pretrained multilingual language models.
* More comprehensive evaluation using BLEU, chrF, and COMET.
* Attention visualization and interpretability analysis.

---

# 📚 References

1. Vaswani, A. et al.
   **Attention Is All You Need.**
   Advances in Neural Information Processing Systems (NeurIPS), 2017.

2. Bahdanau, D., Cho, K. & Bengio, Y.
   **Neural Machine Translation by Jointly Learning to Align and Translate.**
   ICLR, 2015.

3. Sutskever, I., Vinyals, O. & Le, Q. V.
   **Sequence to Sequence Learning with Neural Networks.**
   NeurIPS, 2014.

---

# 👨‍💻 Author

**Praveen Boda**

**B.Tech Electrical Engineering — IIT Madras**

**Course:** DA6401 — Introduction to Deep Learning

---

# ⭐ Project Summary

This project demonstrates the implementation of a **Transformer-based Neural Machine Translation system** using an encoder-decoder architecture.

The model uses **self-attention, multi-head attention, positional encoding, and encoder-decoder cross-attention** to learn relationships between source and target language sequences.

The complete pipeline covers:

```text
Data
 ↓
Tokenization
 ↓
Embeddings + Positional Encoding
 ↓
Transformer Encoder
 ↓
Transformer Decoder
 ↓
Autoregressive Generation
 ↓
Translated Sentence
```

The project provides practical experience with the core ideas behind modern Transformer-based sequence-to-sequence models and Neural Machine Translation.
