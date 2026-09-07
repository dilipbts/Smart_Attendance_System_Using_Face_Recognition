FROM python:3.11-slim

# Configure timezone to IST (Asia/Kolkata)
ENV TZ="Asia/Kolkata"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app"]
