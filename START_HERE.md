# Quickstart Guide: START HERE

Follow these simple steps to run the Lead Conversion web application on your local machine.

---

## Step 1: Install Dependencies

Open a terminal or command prompt in this project folder and run the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Step 2: Ensure ML Artifacts Are Generated

The application relies on a trained model `champion_model.pkl` and metadata. Since they have already been pre-trained and created during setup, you can skip this step. 

If you ever want to re-train the models using the latest raw data in `finance_sales_01.csv`, execute:

```bash
python run_ml_pipeline.py
```

This will retrain Logistic Regression, Random Forest, and XGBoost, rank them, select the best model, and regenerate:
* `champion_model.pkl` (serialized model)
* `feature_importance.csv` (model driver rankings)
* `lead_scores.csv` (predictions for full database)
* `model_results.csv` (leaderboard results)

---

## Step 3: Launch the Application

Run the application server using Python:

```bash
python main.py
```

You should see logs in the terminal indicating that the FastAPI server is running:

```
Starting FastAPI Application Server...
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
Model successfully loaded from: champion_model.pkl
Base data loaded from finance_sales_01.csv with 74,386 records
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Step 4: Open in Your Web Browser

Open your web browser and navigate to:

👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## What to Try in the App

1. **Dashboard tab**: View the key high-level conversion rates, driving features, and lead source efficiency.
2. **Lead Scoring Form tab**: Play with the fields (e.g., set Annual Income to `$150,000`, Credit Score to `750`, Follow-up Consistency to `90`, and Response time to `2` hours). Click **Score Lead** and see the prediction speedometer, priority badge, and recommended action.
3. **Lead Database Browser tab**: Search for a specific Lead ID, or filter by "High Priority" leads. Double click a lead or click **View Details** to open a modal with the lead's entire profile.
4. **Bulk CSV Scorer tab**: Drag and drop a CSV file of new leads (you can test by copying some rows from `finance_sales_01.csv` and removing the `converted` column). Click **Run Scorer**, view the preview, and download the finished scored file.
