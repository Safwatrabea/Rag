FROM python:3.11-slim

WORKDIR /app

# تثبيت الأدوات الضرورية فقط وبناء المكتبات إذا لزم الأمر
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# نسخ المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# نسخ ملفات المشروع
COPY . .

# فتح منفذ التطبيق
EXPOSE 8501

# أمر التشغيل
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
