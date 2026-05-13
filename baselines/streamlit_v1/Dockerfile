FROM python:3.12-slim

# Cyrillic-capable font for PDF report + build deps for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
