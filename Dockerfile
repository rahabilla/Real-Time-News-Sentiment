FROM python:3.10-slim

# Fix: install SSL certificates
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

# Work directory
WORKDIR /app

# Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Streamlit port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "real_time_news_sentiment.py", "--server.port=8501", "--server.address=0.0.0.0"]
