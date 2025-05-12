FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables from .env file at runtime
CMD ["sh", "-c", "if [ -f .env ]; then export $(grep -v '^#' .env | xargs); fi && streamlit run main.py --server.port=8501 --server.address=0.0.0.0"]