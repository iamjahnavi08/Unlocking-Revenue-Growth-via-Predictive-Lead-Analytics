# Production Deployment Guide

This guide describes how to configure, containerize, and deploy the Lead Conversion Analytics application to production environments.

---

## Production Configurations

For development, `main.py` runs Uvicorn with auto-reload. For production deployments:

1. **Turn off reload**: Run without the `--reload` parameter or set `reload=False`.
2. **Increase workers**: Run Uvicorn with multiple worker processes to handle concurrent requests:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```
3. **Environment variables**: Store sensitive database configurations or path changes in environment variables using a `.env` file and retrieve them using python's `os.getenv`.

---

## Containerization with Docker

You can package the application into a lightweight Docker container for uniform deployment across any server environment (AWS, Azure, Google Cloud, DigitalOcean, etc.).

### 1. Create a `Dockerfile`

Create a file named `Dockerfile` in the root of the project with the following content:

```dockerfile
# Use official slim Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (make sure model and datasets are copied or mounted)
COPY main.py .
COPY index.html .
COPY champion_model.pkl .
COPY finance_sales_01.csv .
COPY lead_scores.csv .
COPY feature_importance.csv .
COPY model_results.csv .

# Expose port
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Build and Run the Docker Container

Build the Docker image locally:
```bash
docker build -t lead-predictor-app:latest .
```

Run the container:
```bash
docker run -d -p 8000:8000 --name lead-predictor-service lead-predictor-app:latest
```

Open `http://localhost:8000` in your web browser.

---

## Cloud Deployment Services

### 1. Render / Railway
Render and Railway are simple platforms for deploying Python applications:
- Create a new web service connected to your Git repository.
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 2. AWS Elastic Beanstalk / ECS
- Zip the project files (including the trained `champion_model.pkl`, datasets, and `Dockerfile`).
- Deploy the zip bundle to an AWS Elastic Beanstalk Docker platform, or push the image to AWS ECR and run it inside ECS (Fargate).
