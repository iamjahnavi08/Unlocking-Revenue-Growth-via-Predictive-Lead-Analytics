# Lead Conversion Analytics & Prediction Platform

A production-grade, end-to-end web application that leverages machine learning to predict lead conversion probabilities and score leads in real time. It consists of a **FastAPI backend** that processes predictions using trained machine learning models, and an **interactive HTML5/Tailwind/Chart.js dashboard** that provides visual analytics, a searchable lead database, and a bulk CSV scoring system.

---

## 🌟 Executive Summary & Overview

In today's hyper-competitive business environment, sales teams are overwhelmed with high volumes of inbound leads. Subjective, manual sorting leads to inefficiencies, missed opportunities, and slow response times. This platform transforms raw historical sales data into actionable predictive insights. By analyzing customer demographics, deal parameters, and engagement metrics, the system calculates the exact probability of a lead converting into a closed deal.

Integrating this predictive lead scoring mechanism into sales workflows is projected to deliver a **14% increase in sales revenue** by minimizing outreach latency on high-potential leads.

---

## 🚀 Key Features

1. **Executive Dashboard**: Offers high-level metrics including total leads, historical conversion rates, priority distribution, and interactive charts visualizing top predictive features, lead source conversion efficiency, and product categories conversion.
2. **Real-time Lead Scorer Form**: Allows sales representatives to input a customer's profile (age, credit score, income, debt ratio), deal parameters (value, product category, discount), and engagement metrics (response time, touchpoints, follow-up consistency) to calculate a conversion probability score, lead priority level, and strategic recommendations in real time.
3. **Interactive Database Browser**: A paginated, searchable, and filterable repository of all leads. Sales reps can filter leads by priority (High, Medium, Low) and actual conversion outcome, search by Lead ID, and click to view full profile details in a detailed modal.
4. **Bulk CSV Scorer**: A drag-and-drop file upload zone where users can drop a CSV list of new, unscored leads. The system will engineer features, score them in bulk using the machine learning pipeline, display scoring summary statistics, and provide a downloadable CSV with scores, conversion predictions, and priority categories.

---

## 🛠️ Technology Stack & Methodology

### Core Technology Stack
- **Backend Server**: FastAPI, Uvicorn, Python 3
- **Data Science & ML**: Scikit-Learn, XGBoost, Pandas, Numpy, Joblib, Matplotlib, Seaborn
- **Frontend SPA**: HTML5, Vanilla CSS, Vanilla JS, Tailwind CSS, Chart.js, FontAwesome

### Modeling Methodology
The core machine learning engine evaluates multiple candidate algorithms (Logistic Regression, Random Forest, XGBoost) on historical lead profile data (`finance_sales_01.csv`) and automatically selects the highest-performing champion model based on the **ROC-AUC** metric. Feature engineering incorporates custom, domain-specific heuristics:
- **Normalized Credit Scores**: Bureau credit ratings normalized for statistical linearity.
- **Touchpoints Per Day**: Cumulative interactions normalized by active days in pipeline.
- **Income to Deal Ratios**: Verified customer annual income evaluated against deal contract values.

---

## 📖 Quickstart Guide: Local Setup

Follow these simple steps to run the Lead Conversion web application on your local machine:

### 1. Install Dependencies
Open a terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### 2. Retrain Models (Optional)
The application comes pre-trained with `champion_model.pkl`. If you ever want to re-train the models using the latest raw data in `finance_sales_01.csv`, execute:
```bash
python run_ml_pipeline.py
```
This will evaluate Logistic Regression, Random Forest, and XGBoost, rank them, select the best model, and regenerate model artifacts.

### 3. Launch the Application
Run the application server using Python:
```bash
python main.py
```
This will load the champion model, load the base data, and launch Uvicorn on **`http://127.0.0.1:8000`**.

### 4. Interactive Browser Sandbox
Navigate to **`http://127.0.0.1:8000`** in your browser and try:
- **Dashboard**: Track conversion patterns and key predictive drivers.
- **Lead Scoring Form**: Enter custom profiles to see real-time speedometers and priority badges.
- **Lead Database Browser**: Search by Lead ID or filter leads.
- **Bulk CSV Scorer**: Drag and drop a CSV file of new leads to score them in bulk.

---

## 🐳 Production Deployment Guide

### Uvicorn Production Launch
For production environments, run Uvicorn with auto-reload disabled and multiple worker processes enabled to handle concurrent requests:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Containerization with Docker
Package the application into a lightweight Docker container for uniform deployment across any server environment (AWS, Azure, Google Cloud, Render, etc.).

#### 1. Create a `Dockerfile`
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

# Copy application files
COPY main.py .
COPY index.html .
COPY champion_model.pkl .
COPY finance_sales_01.csv .
COPY lead_scores.csv .
COPY feature_importance.csv .
COPY model_results.csv .

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Build and Run the Docker Image
Build the image:
```bash
docker build -t lead-predictor-app:latest .
```
Run the container:
```bash
docker run -d -p 8000:8000 --name lead-predictor-service lead-predictor-app:latest
```

### Cloud Deployments
- **Render / Railway**: Create a new web service connected to your Git repository. Build command: `pip install -r requirements.txt`, start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **AWS ECS / EKS**: Push the Docker image to AWS ECR and run it inside ECS (Fargate) under an ALB.
