FROM python:3.11-slim

# Force Python logs to print immediately to the PaaS console
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (including libgomp1 for FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose standard PaaS ports
EXPOSE 7860 8080 80

# Use CMD in shell format to handle dynamic $PORT variable from ParsPack
CMD chainlit run app.py --host 0.0.0.0 --port ${PORT:-7860} --headless
