FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the shape predictor model
RUN mkdir -p /app/models && \
    apt-get update && \
    apt-get install -y wget && \
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    apt-get install -y bzip2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 && \
    mv shape_predictor_68_face_landmarks.dat /app/models/ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variable for the predictor path
ENV PREDICTOR_PATH=/app/models/shape_predictor_68_face_landmarks.dat

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 