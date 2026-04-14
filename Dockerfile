FROM python:3.10-slim

# Install modern system dependencies for OpenCV/Video processing
# libgl1 and libglib2.0-0 are the core requirements
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set PYTHONPATH so gunicorn can find 'app.py' inside the 'webapp' folder
ENV PYTHONPATH=/app/webapp

# Expose the port Render uses
EXPOSE 10000

# Start from the root directory but point Gunicorn to the app file
CMD ["gunicorn", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:10000", "webapp.app:app"]
