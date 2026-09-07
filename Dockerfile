FROM python:3.11-slim

ENV TZ="Asia/Kolkata"
ENV DEBIAN_FRONTEND=noninteractive

# Install tzdata and curl (with SSL certificates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    ca-certificates \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download verified models from opencv_zoo with curl
RUN curl -fL -o face_detection_yunet_2023mar.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx && \
    curl -fL -o face_recognition_sface_2021dec.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx && \
    curl -fL -o MiniFASNetV2.onnx \
    https://github.com/face-recognition-anti-spoofing/MiniFASNet/raw/master/saved_models/MiniFASNetV2.onnx || \
    curl -fL -o MiniFASNetV2.onnx \
    https://huggingface.co/qualcomm/MiniFASNet/resolve/main/MiniFASNetV2.onnx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app"]
