FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create data directory
RUN mkdir -p data/users

# Expose port
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
