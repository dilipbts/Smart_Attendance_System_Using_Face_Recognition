FROM python:3.11-slim

# Set container timezone to IST
ENV TZ="Asia/Kolkata"
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal compilers and linear algebra libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    tzdata \
    libopenblas-dev \
    liblapack-dev \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Restrict CMake to 1 core to prevent out-of-memory errors
ENV CMAKE_BUILD_PARALLEL_LEVEL=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app"]
