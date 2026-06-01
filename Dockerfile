# Dockerfile
FROM python:3.11-slim

# Python'un bytecode (.pyc) dosyaları üretmesini ve çıktıları tamponlamasını engelle
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# PyQt5/Qt5'in CLI üzerinde çalışabilmesi veya sistem bağımlılıkları için gerekli kütüphaneleri kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bağımlılıkları kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Varsayılan olarak etkileşimli CLI aracını çalıştır
CMD ["python", "main_cli.py"]
