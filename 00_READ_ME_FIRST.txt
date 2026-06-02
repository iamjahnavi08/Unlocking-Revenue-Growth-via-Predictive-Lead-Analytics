========================================================================
WELCOME TO THE LEAD CONVERSION PREDICTION & ANALYTICS PLATFORM
========================================================================

STATUS: COMPLETE & FULLY OPERATIONAL

The Machine Learning model training pipeline, FastAPI backend APIs, and 
interactive frontend dashboard have all been successfully completed and verified.

------------------------------------------------------------------------
KEY DELIVERABLES ACHIEVED
------------------------------------------------------------------------
1. Machine Learning Pipeline (run_ml_pipeline.py):
   Ingests historical CRM telemetry, performs advanced feature scaling 
   and engineering, trains multiple algorithms, selects the best-performing
   Logistic Regression v2.1 model (ROC-AUC: 81.46%), and exports serializations.

2. FastAPI Backend Server (main.py):
   Fully functional backend exposing high-performance REST APIs to handle
   real-time predictions, bulk CSV scoring, database paging, and executive stats.

3. Frontend Dashboard SPA (index.html):
   An elegant, responsive dark-themed Single Page Application with dynamic 
   interactive charts, real-time prioritizations, and smooth animations.

4. Deployed ML Artifacts:
   * champion_model.pkl (serialized champion model)
   * feature_importance.csv (feature drivers)
   * lead_scores.csv (full scored database)
   * model_results.csv (leaderboard scores)

------------------------------------------------------------------------
QUICK START INSTRUCTIONS
------------------------------------------------------------------------
To launch the application server locally, follow these 3 simple steps:

1. Install Dependencies:
   pip install -r requirements.txt

2. Start the Server:
   python main.py

3. Open in Your Web Browser:
   http://127.0.0.1:8000

Note: For details on containerization, production deployments, and custom 
features, please refer to the unified README.md file in this directory.
========================================================================
