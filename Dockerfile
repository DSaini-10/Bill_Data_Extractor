FROM python:3.11-alpine

# Install Tesseract OCR on Alpine (much faster)
RUN apk add --no-cache \
    tesseract-ocr \
    tesseract-ocr-data-eng

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]