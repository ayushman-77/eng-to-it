FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir aio_pika motor datasets tokenizers tensorboard

# Copy the worker and model code
COPY services/inference_worker/main.py ./
COPY translate.py config.py model.py train.py dataset.py ./
COPY tokenizer_en.json tokenizer_it.json ./

CMD ["python", "main.py"]
