# 🚀 Eng-to-It Translator
**A Distributed Neural Translation Engine**

A professional, microservices-based machine learning inference platform that implements a from-scratch Transformer model (based on *Attention Is All You Need*) for translating English text to Italian. The system features a highly scalable, asynchronous architecture with decoupled API and compute layers.

---

## 📑 Table of Contents
- [✨ Capabilities](#-capabilities)
- [🏗️ Architecture & Pipeline](#️-architecture--pipeline)
- [🧠 Mathematical Formulations](#-mathematical-formulations)
- [📚 Dataset & Tokenization](#-dataset--tokenization)
- [🐳 Docker Deployment (Recommended)](#-docker-deployment-recommended)
- [🚀 Local Setup (Manual)](#-local-setup-manual)
- [🔑 API Reference](#-api-reference)

---

## ✨ Capabilities
This platform is a comprehensive, end-to-end Machine Learning web application capable of:
- **Neural Machine Translation**: High-quality English to Italian translation powered by a custom-built Transformer model.
- **Custom Model Training**: A fully featured training pipeline capable of downloading the Opus Books dataset, building vocabulary tokenizers, and training the transformer architecture from scratch on CPU or GPU (compatible with Kaggle and Google Colab).
- **Asynchronous Task Processing**: Decoupled architecture using RabbitMQ to ensure that heavy ML inference tasks do not block the web API.
- **Scalable Compute Layer**: A dedicated High-Performance C++ Inference Worker that pulls jobs from the message queue, performs hardware-accelerated translation via LibTorch/ONNX, and updates the database.
- **Secure REST API**: A FastAPI-based API Gateway featuring JWT (JSON Web Token) authentication, user registration, and secure endpoints.
- **Persistent Storage**: MongoDB integration to securely store user credentials, track translation job states (pending, processing, completed), and maintain translation history.

---

## 🏗️ Architecture & Pipeline
The system is designed with modern microservice patterns to ensure responsiveness and scalability:
1. **Client Request**: A user authenticates via the `/api/v1/auth/login` endpoint to receive a secure JWT token.
2. **Job Submission**: The client submits English text to the API Gateway (`POST /api/v1/translate/`). 
3. **Queueing**: The API Gateway validates the token, registers a `pending` translation job in MongoDB, publishes the job details to a RabbitMQ message exchange, and immediately returns a `202 Accepted` response with a unique Job ID.
4. **Inference Worker**: The background High-Performance C++ worker consumes the message from RabbitMQ. It updates the job status to `processing`, loads the pre-trained model into LibTorch execution graphs, and executes the highly-optimized Transformer decoding loop.
5. **Completion**: The C++ worker saves the final Italian translation back to MongoDB via mongocxx, marking the job as `completed`, and acknowledges the message in RabbitMQ.
6. **Result Retrieval**: The client polls the API (`GET /api/v1/translate/{job_id}`) to retrieve the finished translation, or accesses their full translation history via the React frontend.

---

## 🧠 Mathematical Formulations
The ML model implements the complete **Sequence-to-Sequence Encoder-Decoder Transformer** pipeline:
1. **Scaled Dot-Product Attention:**
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
2. **Multi-Head Attention:**
   $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
3. **Sinusoidal Positional Encoding:**
   $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
4. **Causal Masking (Decoder):**
   Upper-triangular masking ensures token at index $t$ can only attend to previous positions $\le t$.

---

## 📚 Dataset & Tokenization
* **Dataset:** [`Helsinki-NLP/opus_books`](https://huggingface.co/datasets/Helsinki-NLP/opus_books) (English $\rightarrow$ Italian translation pairs from classical literature).
* **Tokenizer:** Custom `WordLevel` tokenizers trained dynamically using Hugging Face `tokenizers` library with special tokens (`[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`).

---

## 🐳 Docker Deployment (Recommended)
This application is fully containerized. You can spin up the entire microservices architecture (MongoDB, RabbitMQ, API Gateway, PyTorch Worker, and React Frontend) with a single command on any machine.

1. Make sure you have Docker and Docker Compose installed.
2. Ensure your trained PyTorch weights are placed in the `weights/` directory (e.g., `weights/tmodel_19.pt`).
3. Run the following command from the project root:
```bash
docker-compose up --build
```
This will automatically build and launch all 5 containers securely linked via an internal Docker network. 
- The Frontend will be served at `http://localhost:5173`
- The API Swagger UI will be at `http://localhost:8000/docs`

---

## 🚀 Local Setup (Manual)

### 1. Start Infrastructure
Ensure MongoDB (`mongod`) and RabbitMQ (`rabbitmq-server`) are running in the background. Place your weights in the `weights/` directory.

### 2. Terminal 1: API Gateway
```bash
cd services/api_gateway
source .venv/Scripts/activate
pip install -r requirements.txt
python -m app.main
```

### 3. Terminal 2: Inference Worker
Open a new terminal in the global Python environment (with PyTorch installed):
```bash
pip install aio_pika motor
python python_worker.py
```

### 4. Terminal 3: React Frontend
```bash
cd services/frontend
npm install
npm run dev
```

---

## 🔑 API Reference
All translation endpoints require a `Bearer` token obtained from `/api/v1/auth/login`.

| Method | Endpoint                        | Auth     | Description                          |
| :----- | :------------------------------ | :------- | :----------------------------------- |
| POST   | `/api/v1/auth/register`         | —        | Create a new user account            |
| POST   | `/api/v1/auth/login`            | —        | Authenticate, receive JWT            |
| POST   | `/api/v1/translate/`            | Bearer   | Submit English text for translation  |
| GET    | `/api/v1/translate/{job_id}`    | Bearer   | Poll job status / retrieve result    |

---

## 📖 References
* Vaswani, A., et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS.
* Hugging Face `datasets` & `tokenizers`.
* [FastAPI Documentation](https://fastapi.tiangolo.com/)
* [RabbitMQ Documentation](https://www.rabbitmq.com/)
