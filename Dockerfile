FROM python:3.11-slim

# Install build tools, CMake, and C++ dependencies for dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render provides PORT environment variable dynamically (defaults to 10000)
ENV PORT=10000
EXPOSE 10000

# Replace 'app:app' with 'your_filename:app' if your main file isn't app.py
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT app:app"]
