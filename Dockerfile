# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app folder
COPY ./app /app

# Expose port (optional, Cloud Run ignores this)
EXPOSE 80

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]