# Lead Conversion Analytics & Prediction Platform

A production-grade, end-to-end web application that leverages machine learning to predict lead conversion probabilities and score leads in real time. It consists of a **FastAPI backend** that processes predictions using trained machine learning models, and an **interactive HTML5/Tailwind/Chart.js dashboard** that provides visual analytics, a searchable lead database, and a bulk CSV scoring system.

## Key Features

1. **Executive Dashboard**: Offers high-level metrics including total leads, historical conversion rates, priority distribution, and interactive charts visualizing top predictive features, lead source conversion efficiency, and product categories conversion.
2. **Real-time Lead Scorer Form**: Allows sales representatives to input a customer's profile (age, credit score, income, debt ratio), deal parameters (value, product category, discount), and engagement metrics (response time, touchpoints, follow-up consistency) to calculate a conversion probability score, lead priority level, and strategic recommendations in real time.
3. **Interactive Database Browser**: A paginated, searchable, and filterable repository of all leads. Sales reps can filter leads by priority (High, Medium, Low) and actual conversion outcome, search by Lead ID, and click to view full profile details in a detailed modal.
4. **Bulk CSV Scorer**: A drag-and-drop file upload zone where users can drop a CSV list of new, unscored leads. The system will engineer features, score them in bulk using the machine learning pipeline, display scoring summary statistics, and provide a downloadable CSV with scores, conversion predictions, and priority categories.

## Architecture

- **Machine Learning Pipeline (`run_ml_pipeline.py`)**: A clean Python CLI workflow that loads raw historical data (`finance_sales_01.csv`), performs feature engineering, trains candidate classifiers (Logistic Regression, Random Forest, XGBoost), selects the highest-performing champion model by ROC-AUC, and exports serializations and statistics.
- **Backend APIs (`main.py`)**: A FastAPI server that loads the champion model (`champion_model.pkl`) and exposes endpoints to fetch analytics (`/api/stats`), list leads (`/api/leads`), perform single inference (`/api/predict`), and run bulk inference (`/api/bulk-predict`).
- **Frontend SPA (`index.html`)**: A self-contained single-page responsive user interface with an elegant dark theme, charts powered by Chart.js, and styled using Tailwind CSS and FontAwesome.

## Files Created

- `main.py`: The entry point for running the web application.
- `index.html`: The HTML structure and JavaScript runtime for the user interface.
- `run_ml_pipeline.py`: Replicable script to run data processing and model retraining.
- `requirements.txt`: Python package dependencies.
- `README.md`: General description of the platform.
- `START_HERE.md`: Step-by-step instructions on starting and using the application.
- `DEPLOYMENT_GUIDE.md`: Guidelines for production deployment and containerization.
