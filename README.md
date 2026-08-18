# Transformer from Scratch in PyTorch (English to Italian)

An implementation of the original **Encoder-Decoder Transformer** neural network from the seminal research paper [*"Attention Is All You Need"*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), written in pure **PyTorch**.

This repository builds every single Transformer component from first principles without relying on high-level pre-built modules like `nn.Transformer`. It includes complete training, validation, autoregressive greedy decoding, interactive translation, and multi-head attention visualization.

---

## Table of Contents
- [Architecture Overview](#-architecture-overview)
  - [Mathematical Formulations](#mathematical-formulations)
- [Project Structure](#-project-structure)
- [Dataset & Tokenization](#-dataset--tokenization)
- [Installation & Setup](#-installation--setup)
- [Training](#-training)
  - [1. Local Training (CPU)](#1-local-training-cpu)
  - [2. Cloud Training (Google Colab GPU)](#2-cloud-training-google-colab-gpu)
- [Inference & Translation](#-inference--translation)
- [Attention Visualization](#-attention-visualization)
- [Hyperparameters](#-hyperparameters)
- [References](#-references)

## Architecture & Workflow Overview

The model implements the complete **Sequence-to-Sequence Encoder-Decoder Transformer** pipeline:

1. **Input Embedding & Positional Encoding:** Source English and target Italian tokens are projected into continuous dense vectors ($d_{model}$) and summed with sinusoidal positional encodings to preserve word order.
2. **Encoder Stack ($N$ Layers):** Processes the source sentence bidirectionally through stacked Multi-Head Self-Attention and Position-wise Feed-Forward layers, applying Layer Normalization and Residual Additive connections at each sub-layer.
3. **Decoder Stack ($N$ Layers):** Autoregressively generates target tokens using:
   - **Masked Self-Attention:** Applies a causal lower-triangular mask to prevent future token lookahead.
   - **Cross-Attention:** Queries the final encoder output representations to dynamically align source English words with target Italian tokens.
   - **Feed-Forward Sub-layer:** Applies non-linear point-wise feature projections.
4. **Linear Projection & Output:** Maps decoder hidden states to target vocabulary logits to compute Cross-Entropy loss during training or select next-token predictions via greedy decoding during inference.

---

### Mathematical Formulations

1. **Scaled Dot-Product Attention:**
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

2. **Multi-Head Attention:**
   $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
   $$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

3. **Sinusoidal Positional Encoding:**
   $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
   $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

4. **Position-wise Feed-Forward Network:**
   $$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

5. **Layer Normalization:**
   $$\text{LayerNorm}(x) = \alpha \odot \left(\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \beta$$

6. **Residual Additive Connections:**
   $$\text{Output} = x + \text{Dropout}(\text{SubLayer}(\text{LayerNorm}(x)))$$

7. **Causal Masking (Decoder):**
   Upper-triangular masking ensures token at index $t$ can only attend to previous positions $\le t$.

---

## Project Structure

```text
eng-to-it/
├── model.py                # Core Transformer architecture built from scratch
├── dataset.py              # Bilingual dataset loader, padding & causal masking
├── config.py               # Hyperparameters and checkpoint directory paths
├── train.py                # Training loop, validation, TensorBoard logging & greedy decode
├── translate.py            # CLI & programmatic inference for translating sentences
├── eng2it.ipynb            # Google Colab GPU training notebook
├── attention_visual.ipynb  # Interactive Altair attention heatmaps visualization
├── .gitignore              # Ignores heavy weights, venvs, and cache files
└── README.md               # Project documentation
```

---

## Dataset & Tokenization

* **Dataset:** [`Helsinki-NLP/opus_books`](https://huggingface.co/datasets/Helsinki-NLP/opus_books) (English $\rightarrow$ Italian translation pairs from classical literature).
* **Tokenizer:** Custom `WordLevel` tokenizers trained dynamically using Hugging Face `tokenizers` library with special tokens:
  * `[UNK]` — Unknown Token (Index 0)
  * `[PAD]` — Padding Token (Index 1)
  * `[SOS]` — Start of Sentence (Index 2)
  * `[EOS]` — End of Sentence (Index 3)

---

## 🛠 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ayushman-77/eng-to-it.git
   cd eng-to-it
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install datasets tokenizers tensorboard tqdm altair pandas numpy
   ```

---

## Training

### 1. Local Training (CPU)
Run the training script locally:
```bash
python train.py
```
* Tokenizers will automatically be built and saved on the first run (`tokenizer_en.json`, `tokenizer_it.json`).
* Checkpoints are saved at the end of each epoch into the `weights/` folder (e.g., `weights/tmodel_00.pt`).

### 2. Cloud Training (Google Colab GPU)
For fast GPU training (~1–2 minutes per epoch):
1. Open the included [`eng2it.ipynb`](eng2it.ipynb) in [Google Colab](https://colab.research.google.com).
2. Set Runtime to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
3. Connect your Google Drive to save checkpoints permanently and run all cells.

---

## Inference & Translation

Translate any sentence from the command line:

```bash
python translate.py "I am reading a book in my room."
```

Output:
```text
Input sentence: I am reading a book in my room.
Loaded weights from weights/tmodel_19.pt
Translation (it): Sono leggendo un libro nella mia stanza .
```

Or use the Python API in your own code:
```python
from translate import translate

result = translate("He walked towards the house.")
print(result) # "Camminava verso la casa ."
```

---

## Attention Visualization

Open [`attention_visual.ipynb`](attention_visual.ipynb) to inspect the learned attention weights:
* **Encoder Self-Attention:** Shows which source words attend to other source words.
* **Decoder Masked Self-Attention:** Visualizes autoregressive target word dependencies.
* **Encoder-Decoder Cross-Attention:** Shows word alignments between English and Italian across all 8 attention heads.

---

## Hyperparameters

Default configuration defined in [`config.py`](config.py):

| Hyperparameter | Default (Fast/CPU) | Full Paper (GPU) | Description |
| :--- | :--- | :--- | :--- |
| **`d_model`** | `256` | `512` | Embedding & hidden representation dimension |
| **`N`** | `3` | `6` | Number of Encoder and Decoder layers |
| **`h`** | `8` | `8` | Number of Multi-Head Attention heads |
| **`d_ff`** | `512` | `2048` | Dimension of Feed-Forward hidden layer |
| **`seq_len`** | `80` | `350` | Maximum sentence token length |
| **`batch_size`** | `32` | `32` / `64` | Batch size during training |
| **`lr`** | `3e-4` | `1e-4` | Adam optimizer learning rate |
| **`dropout`** | `0.1` | `0.1` | Dropout rate for regularization |
| **`label_smoothing`** | `0.1` | `0.1` | Label smoothing for CrossEntropyLoss |

---

## References
* Vaswani, A., et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). Advances in Neural Information Processing Systems (NeurIPS).
* Hugging Face [`datasets`](https://huggingface.co/docs/datasets) & [`tokenizers`](https://huggingface.co/docs/tokenizers).
* OPUS Books Dataset: [Tiedemann, J. (2012). *Parallel Data, Tools and Interfaces in OPUS*](https://opus.nlpl.eu/).

---

## License
MIT License. Feel free to use, modify, and distribute for educational and research purposes.
