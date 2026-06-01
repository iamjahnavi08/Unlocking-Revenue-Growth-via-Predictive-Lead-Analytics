import os
import io
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn

app = FastAPI(
    title="Lead Conversion Analytics Platform",
    description="Real-time Lead Scoring & Predictive Analytics App",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
MODEL_PATH = "champion_model.pkl"
DATA_PATH = "finance_sales_01.csv"
SCORES_PATH = "lead_scores.csv"
IMPORTANCE_PATH = "feature_importance.csv"
RESULTS_PATH = "model_results.csv"

model_pipeline = None
df_full_leads = None
feature_defaults = {}
median_annual_income = 81831.0
unique_categories = {}

# Predefined columns lists
NUMERICAL_COLS = [
    'deal_value_usd', 'customer_age', 'customer_annual_income', 'customer_credit_score',
    'customer_existing_products_count', 'customer_relationship_tenure_years',
    'customer_debt_to_income_ratio', 'sales_rep_experience_years', 'sales_rep_monthly_quota_usd',
    'sales_rep_ytd_quota_attainment_pct', 'first_contact_response_hrs', 'days_in_pipeline',
    'num_total_touchpoints', 'num_calls_made', 'num_emails_sent', 'num_meetings_held',
    'proposal_sent', 'needs_analysis_completed', 'competitor_present', 'discount_offered_pct',
    'manager_escalated', 'customer_objection_raised', 'follow_up_consistency_score'
]

CATEGORICAL_COLS = [
    'lead_source', 'product_category', 'product_sub_category', 'customer_type',
    'customer_gender', 'customer_employment_type', 'customer_state', 'customer_city_tier',
    'sales_rep_region', 'sales_rep_tier', 'last_stage_reached', 'sentiment_last_interaction'
]

@app.on_event("startup")
def startup_event():
    global model_pipeline, df_full_leads, feature_defaults, median_annual_income, unique_categories
    
    # 1. Load Model
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            print(f"Model successfully loaded from: {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found. Prediction endpoints will fail until the model is trained.")
        
    # 2. Load Datasets
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            print(f"Base data loaded from {DATA_PATH} with {len(df):,} records")
            
            # Load scores
            if os.path.exists(SCORES_PATH):
                df_scores = pd.read_csv(SCORES_PATH)
                print(f"Scores loaded from {SCORES_PATH}")
                # Merge base data with lead scores
                df_full_leads = pd.merge(df, df_scores, on='lead_id', how='left')
            else:
                print(f"Scores file {SCORES_PATH} not found. Scoring full dataset inline...")
                df_full_leads = df.copy()
                df_full_leads['lead_score'] = 0.0
                df_full_leads['predicted_conversion'] = 0
                df_full_leads['lead_category'] = 'Unscored'
                
            # Fill missing scores with defaults
            df_full_leads['lead_score'] = df_full_leads['lead_score'].fillna(0.0)
            df_full_leads['predicted_conversion'] = df_full_leads['predicted_conversion'].fillna(0).astype(int)
            df_full_leads['lead_category'] = df_full_leads['lead_category'].fillna('Low Priority')
            
            # Compute defaults for real-time predictor
            median_annual_income = float(df['customer_annual_income'].median())
            
            # Numerical defaults (median values)
            for col in NUMERICAL_COLS:
                if col in df.columns:
                    feature_defaults[col] = float(df[col].median())
                else:
                    feature_defaults[col] = 0.0
            
            # Categorical defaults (mode or first category)
            for col in CATEGORICAL_COLS:
                if col in df.columns:
                    unique_categories[col] = df[col].dropna().unique().tolist()
                    feature_defaults[col] = df[col].mode()[0] if not df[col].mode().empty else ""
                else:
                    unique_categories[col] = []
                    feature_defaults[col] = ""
                    
            print("Feature defaults and category schemas initialized.")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            df_full_leads = pd.DataFrame()
    else:
        print(f"Warning: Base dataset '{DATA_PATH}' not found. App analytics and DB view will be empty.")
        df_full_leads = pd.DataFrame()

# Pydantic Model for real-time lead prediction input
class LeadInput(BaseModel):
    deal_value_usd: Optional[float] = None
    customer_age: Optional[int] = None
    customer_annual_income: Optional[float] = None
    customer_credit_score: Optional[int] = None
    customer_existing_products_count: Optional[int] = None
    customer_relationship_tenure_years: Optional[float] = None
    customer_debt_to_income_ratio: Optional[float] = None
    sales_rep_experience_years: Optional[float] = None
    sales_rep_monthly_quota_usd: Optional[float] = None
    sales_rep_ytd_quota_attainment_pct: Optional[float] = None
    first_contact_response_hrs: Optional[float] = None
    days_in_pipeline: Optional[int] = None
    num_total_touchpoints: Optional[int] = None
    num_calls_made: Optional[int] = None
    num_emails_sent: Optional[int] = None
    num_meetings_held: Optional[int] = None
    proposal_sent: Optional[int] = None
    needs_analysis_completed: Optional[int] = None
    competitor_present: Optional[int] = None
    discount_offered_pct: Optional[float] = None
    manager_escalated: Optional[int] = None
    customer_objection_raised: Optional[int] = None
    follow_up_consistency_score: Optional[int] = None
    
    lead_source: Optional[str] = None
    product_category: Optional[str] = None
    product_sub_category: Optional[str] = None
    customer_type: Optional[str] = None
    customer_gender: Optional[str] = None
    customer_employment_type: Optional[str] = None
    customer_state: Optional[str] = None
    customer_city_tier: Optional[str] = None
    sales_rep_region: Optional[str] = None
    sales_rep_tier: Optional[str] = None
    last_stage_reached: Optional[str] = None
    sentiment_last_interaction: Optional[str] = None

# 1. API: Single Predict Endpoint
@app.post("/api/predict")
def predict_single(lead_input: LeadInput):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="ML Model not loaded on server. Please train the model.")
        
    # Convert input to dictionary
    input_dict = lead_input.dict()
    
    # Fill in missing values with pre-calculated defaults
    final_dict = {}
    for col in NUMERICAL_COLS:
        final_dict[col] = input_dict[col] if input_dict[col] is not None else feature_defaults.get(col, 0.0)
    for col in CATEGORICAL_COLS:
        final_dict[col] = input_dict[col] if input_dict[col] is not None else feature_defaults.get(col, "")
        
    # Perform Feature Engineering
    final_dict['income_to_deal_ratio'] = final_dict['customer_annual_income'] / (final_dict['deal_value_usd'] + 1)
    final_dict['touchpoints_per_day'] = final_dict['num_total_touchpoints'] / (final_dict['days_in_pipeline'] + 1)
    final_dict['credit_score_normalized'] = final_dict['customer_credit_score'] / 850
    final_dict['high_value_customer'] = int(final_dict['customer_annual_income'] > median_annual_income)
    
    # Convert to DataFrame (ensure proper formatting)
    df_input = pd.DataFrame([final_dict])
    
    try:
        # Run prediction
        prob = float(model_pipeline.predict_proba(df_input)[0][1])
        pred = int(model_pipeline.predict(df_input)[0])
        
        # Categorize
        if prob >= 0.70:
            category = 'High Priority'
            color = 'purple'
            rec = "Assign immediately to a Senior Sales Rep. Ensure immediate response within 2 hours. Offer product demonstration."
        elif prob >= 0.30:
            category = 'Medium Priority'
            color = 'yellow'
            rec = "Follow up consistently via email and calls. Provide product brochure and schedule needs analysis call."
        else:
            category = 'Low Priority'
            color = 'blue'
            rec = "Add to automated nurture email campaigns. Monitor for digital interaction touchpoints before manual follow-up."
            
        # Explanations based on key inputs
        key_factors = []
        if final_dict['follow_up_consistency_score'] < 50:
            key_factors.append("Low follow-up consistency score limits conversion odds.")
        elif final_dict['follow_up_consistency_score'] > 75:
            key_factors.append("High follow-up consistency score strongly drives conversion.")
            
        if final_dict['first_contact_response_hrs'] > 24:
            key_factors.append("Slow first contact response time (>24 hrs) negatively impacts lead warmth.")
        elif final_dict['first_contact_response_hrs'] < 4:
            key_factors.append("Lightning fast response time (<4 hrs) substantially boosts engagement.")
            
        if final_dict['customer_credit_score'] < 600:
            key_factors.append("Customer's sub-prime credit score (<600) increases risk & lowers approval rate.")
        elif final_dict['customer_credit_score'] >= 720:
            key_factors.append("Excellent credit score (>720) represents a low-risk, high-quality opportunity.")
            
        if len(key_factors) == 0:
            key_factors.append("Lead demonstrates standard characteristics with moderate conversion probability.")
            
        return {
            "success": True,
            "lead_score": prob,
            "prediction": pred,
            "lead_category": category,
            "color": color,
            "recommendation": rec,
            "key_factors": key_factors,
            "input_used": final_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# 2. API: Leads List (Database View) Endpoint
@app.get("/api/leads")
def get_leads(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = None,
    priority: Optional[str] = None,
    converted: Optional[int] = None,
    sort_by: Optional[str] = 'lead_score',
    sort_desc: bool = True
):
    global df_full_leads
    if df_full_leads is None or df_full_leads.empty:
        return {"leads": [], "total": 0, "page": page, "limit": limit, "pages": 0}
        
    filtered_df = df_full_leads.copy()
    
    # Filter by search (lead_id)
    if search:
        filtered_df = filtered_df[filtered_df['lead_id'].str.contains(search, case=False, na=False)]
        
    # Filter by priority
    if priority and priority != 'All':
        filtered_df = filtered_df[filtered_df['lead_category'] == priority]
        
    # Filter by conversion status
    if converted is not None:
        filtered_df = filtered_df[filtered_df['converted'] == converted]
        
    # Sort
    if sort_by and sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=not sort_desc)
        
    total_count = len(filtered_df)
    total_pages = (total_count + limit - 1) // limit
    
    # Slice for pagination
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_df = filtered_df.iloc[start_idx:end_idx]
    
    # Convert to JSON serializable objects
    leads_list = paginated_df.to_dict(orient='records')
    
    # Clean NaN values for JSON output
    for lead in leads_list:
        for key, val in lead.items():
            if isinstance(val, float) and np.isnan(val):
                lead[key] = None
                
    return {
        "leads": leads_list,
        "total": total_count,
        "page": page,
        "limit": limit,
        "pages": total_pages
    }

# 3. API: Statistics Dashboard Endpoint
@app.get("/api/stats")
def get_stats():
    global df_full_leads
    
    # Load feature importance if exists
    feature_importance = []
    if os.path.exists(IMPORTANCE_PATH):
        try:
            df_imp = pd.read_csv(IMPORTANCE_PATH)
            # Take top 10
            feature_importance = df_imp.head(10).to_dict(orient='records')
        except Exception as e:
            print(f"Error loading feature importance: {e}")
            
    # Load model results if exists
    model_results = []
    if os.path.exists(RESULTS_PATH):
        try:
            df_res = pd.read_csv(RESULTS_PATH)
            model_results = df_res.to_dict(orient='records')
        except Exception as e:
            print(f"Error loading model results: {e}")
            
    if df_full_leads is None or df_full_leads.empty:
        return {
            "summary": {
                "total_leads": 0,
                "conversion_rate": 0.0,
                "high_priority_pct": 0.0,
                "avg_lead_score": 0.0,
                "total_deal_value": 0.0,
                "projected_deal_value": 0.0
            },
            "lead_sources": [],
            "product_categories": [],
            "feature_importance": feature_importance,
            "model_results": model_results,
            "category_schemas": unique_categories
        }
        
    # Calculate Summary Stats
    total_leads = len(df_full_leads)
    conversion_rate = float(df_full_leads['converted'].mean() * 100) if 'converted' in df_full_leads.columns else 0.0
    
    high_priority_count = len(df_full_leads[df_full_leads['lead_category'] == 'High Priority'])
    high_priority_pct = float(high_priority_count / total_leads * 100) if total_leads > 0 else 0.0
    
    avg_score = float(df_full_leads['lead_score'].mean() * 100)
    
    # Calculate aggregate pipeline values
    total_deal_val = 0.0
    projected_deal_val = 0.0
    if 'deal_value_usd' in df_full_leads.columns:
        total_deal_val = float(df_full_leads['deal_value_usd'].sum())
        if 'lead_score' in df_full_leads.columns:
            projected_deal_val = float((df_full_leads['deal_value_usd'] * df_full_leads['lead_score']).sum())
            
    # Lead Source Analytics
    source_stats = []
    if 'lead_source' in df_full_leads.columns and 'converted' in df_full_leads.columns:
        source_grouped = df_full_leads.groupby('lead_source')['converted'].agg(['count', 'mean']).reset_index()
        source_grouped['mean'] = source_grouped['mean'] * 100
        source_grouped = source_grouped.sort_values(by='mean', ascending=False)
        source_stats = source_grouped.to_dict(orient='records')
        
    # Product Category Analytics
    product_stats = []
    if 'product_category' in df_full_leads.columns and 'converted' in df_full_leads.columns:
        prod_grouped = df_full_leads.groupby('product_category')['converted'].agg(['count', 'mean']).reset_index()
        prod_grouped['mean'] = prod_grouped['mean'] * 100
        prod_grouped = prod_grouped.sort_values(by='mean', ascending=False)
        product_stats = prod_grouped.to_dict(orient='records')
        
    return {
        "summary": {
            "total_leads": total_leads,
            "conversion_rate": round(conversion_rate, 2),
            "high_priority_pct": round(high_priority_pct, 2),
            "avg_lead_score": round(avg_score, 2),
            "total_deal_value": round(total_deal_val, 2),
            "projected_deal_value": round(projected_deal_val, 2)
        },
        "lead_sources": source_stats,
        "product_categories": product_stats,
        "feature_importance": feature_importance,
        "model_results": model_results,
        "category_schemas": unique_categories
    }

# 4. API: Bulk Scoring CSV Upload Endpoint
@app.post("/api/bulk-predict")
async def bulk_predict(file: UploadFile = File(...)):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="ML Model not loaded on server. Please train the model.")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = await file.read()
        df_uploaded = pd.read_csv(io.BytesIO(contents))
        
        # Verify columns or fill missing
        missing_cols = []
        for col in NUMERICAL_COLS + CATEGORICAL_COLS:
            if col not in df_uploaded.columns:
                missing_cols.append(col)
                # Pad missing column with default value
                df_uploaded[col] = feature_defaults.get(col)
                
        # Perform Feature Engineering on Uploaded CSV
        df_uploaded['income_to_deal_ratio'] = df_uploaded['customer_annual_income'] / (df_uploaded['deal_value_usd'] + 1)
        df_uploaded['touchpoints_per_day'] = df_uploaded['num_total_touchpoints'] / (df_uploaded['days_in_pipeline'] + 1)
        df_uploaded['credit_score_normalized'] = df_uploaded['customer_credit_score'] / 850
        df_uploaded['high_value_customer'] = (df_uploaded['customer_annual_income'] > median_annual_income).astype(int)
        
        # Predict
        lead_scores = model_pipeline.predict_proba(df_uploaded)[:, 1]
        predictions = model_pipeline.predict(df_uploaded)
        
        # Add columns
        df_uploaded['lead_score'] = lead_scores
        df_uploaded['predicted_conversion'] = predictions
        
        # Determine priority category
        df_uploaded['lead_category'] = pd.cut(
            df_uploaded['lead_score'],
            bins=[0.0, 0.3, 0.7, 1.0],
            labels=['Low Priority', 'Medium Priority', 'High Priority'],
            include_lowest=True
        )
        
        # Select first 20 records for JSON preview
        preview_cols = ['lead_id', 'customer_annual_income', 'deal_value_usd', 'lead_score', 'predicted_conversion', 'lead_category']
        available_preview_cols = [c for c in preview_cols if c in df_uploaded.columns]
        preview_data = df_uploaded[available_preview_cols].head(20).to_dict(orient='records')
        
        # Clean NaNs in preview
        for row in preview_data:
            for k, v in row.items():
                if isinstance(v, float) and np.isnan(v):
                    row[k] = None
                    
        # Export full scored CSV to string buffer
        out_buf = io.StringIO()
        df_uploaded.to_csv(out_buf, index=False)
        out_buf.seek(0)
        
        # Convert string buffer to bytes response
        csv_bytes = out_buf.getvalue().encode('utf-8')
        
        summary_stats = {
            "total_scored": len(df_uploaded),
            "predicted_conversions": int(df_uploaded['predicted_conversion'].sum()),
            "conversion_rate_pct": float(df_uploaded['predicted_conversion'].mean() * 100),
            "high_priority_pct": float((df_uploaded['lead_category'] == 'High Priority').mean() * 100)
        }
        
        # Return summary and file preview. Frontend can download binary later.
        # To download, we provide a separate route or serve as base64 or stream
        # Let's save the last scored CSV in server memory or temp file for retrieval
        # Or even better, return JSON with base64 encoded CSV so the frontend can download it instantly with one API request!
        import base64
        csv_b64 = base64.b64encode(csv_bytes).decode('utf-8')
        
        return {
            "success": True,
            "summary": summary_stats,
            "preview": preview_data,
            "csv_data": csv_b64,
            "filename": f"scored_{file.filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk processing error: {str(e)}")

# 5. HTML Frontend Route
@app.get("/", response_class=HTMLResponse)
def read_root():
    # Read index.html
    html_path = "index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        <html>
            <head><title>App Error</title></head>
            <body style="font-family: sans-serif; padding: 50px; text-align: center;">
                <h2>index.html is missing!</h2>
                <p>Please place index.html in the same directory as main.py.</p>
            </body>
        </html>
        """

if __name__ == "__main__":
    print("Starting FastAPI Application Server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
