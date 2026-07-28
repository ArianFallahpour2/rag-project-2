# 1. Base Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install C++ runtime libraries (libgomp1 is required by FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all project files into the container
COPY . .

# 6. Expose port 7860 for ParsPack PaaS routing
EXPOSE 7860

# 7. Start the Chainlit application
ENTRYPOINT ["chainlit", "run", "app.py", "-h", "--host", "0.0.0.0", "--port", "7860"]
