# Project Summary: Lead Conversion Prediction Platform

## Overview
This platform transforms raw historical sales data into actionable predictive insights. By analyzing customer demographics, deal parameters, and engagement metrics, the system calculates the exact probability of a lead converting into a closed deal.

## Methodology
The core machine learning engine evaluates multiple algorithms (Logistic Regression, Random Forest, XGBoost) and automatically selects the highest performing model (Champion Model) based on the ROC-AUC metric. Feature engineering incorporates domain-specific heuristics such as:
- Normalized Credit Scores
- Touchpoints Per Day
- Income to Deal Ratios

## Technology Stack
- **Backend**: FastAPI, Uvicorn, Python 3
- **Data Science**: Scikit-Learn, XGBoost, Pandas, Numpy, Joblib
- **Frontend**: HTML5, Tailwind CSS, Chart.js

## Outcome
The final application successfully unifies the machine learning workflow with an elegant user interface, allowing sales representatives and managers to upload bulk data or interactively score leads in real-time to prioritize their daily activities.
