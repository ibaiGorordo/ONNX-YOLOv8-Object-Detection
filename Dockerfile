FROM python:3.10

RUN apt update && \
    apt install -y htop libgl1-mesa-glx libglib2.0-0
RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY . onnx

