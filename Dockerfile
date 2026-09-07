FROM python:3.11-slim

ENV TZ="Asia/Kolkata"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    wget \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Auto-download detection, recognition, and anti-spoofing models
RUN wget -q -O face_detection_yunet_2023mar.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx && \
    wget -q -O face_recognition_sface_2021dec.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx && \
    wget -q -O MiniFASNetV2.onnx \
    https://raw.githubusercontent.com/computervisioneng/face-anti-spoofing-onnx/main/models/MiniFASNetV2.onnx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app"]
