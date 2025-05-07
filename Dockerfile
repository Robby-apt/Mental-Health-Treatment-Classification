# Use an official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependencies file
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code into the container
COPY . .

# Expose the port your app runs on (usually 5000)
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]
