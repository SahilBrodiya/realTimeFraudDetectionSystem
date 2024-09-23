# Use Python slim image as base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file into container
COPY requirements.txt requirements.txt

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files into container
COPY . .

# Command to run when starting the container
CMD ["python", "app.py"]