import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pickle


sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11

print("  Libraries imported successfully!")


csv_path = "finance_sales_01(2).csv"
if not os.path.exists(csv_path):
    csv_path =  r'C:\Users\akumalla.jahnavi.EXAFLUENCE-INC\Desktop\testing app project\finance_sales_01.csv'



df = pd.read_csv(csv_path)

print(f"Dataset successfully loaded")
print(f" Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n Column Names:\n{df.columns.tolist()}")
print(f"\n Data Types:\n{df.dtypes}")
print(f"\n Missing Values:\n{df.isnull().sum().sum()} total null values")
print(f"\n First 5 Rows:")
df.head()


print("Statistical Summary:")
df.describe()


print("Target Variable Distribution:")
print(df['converted'].value_counts())
print(f"\nConversion Rate: {df['converted'].mean()*100:.2f}%")


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

colors = ["#acff9985", '#66b3ff']
converted_counts = df['converted'].value_counts()
axes[0].pie(converted_counts.values, labels=['Not Converted', 'Converted'], autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
axes[0].set_title('Lead Conversion Balance', fontsize=14, fontweight='bold')

sns.countplot(x='converted', hue='converted', data=df, palette='Set2', ax=axes[1], legend=False)
axes[1].set_title('Lead Conversion Count', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Converted (0 = No, 1 = Yes)', fontsize=11)
axes[1].set_ylabel('Number of Leads', fontsize=11)
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height()):,}\n({p.get_height()/len(df)*100:.1f}%)', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

features_to_plot = ['customer_credit_score', 'customer_annual_income', 
                     'customer_relationship_tenure_years', 'first_contact_response_hrs']

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx // 2, idx % 2]
    sns.boxplot(x='converted', y=feature, hue='converted', data=df, palette='Set2', ax=ax, legend=False)
    ax.set_title(f'{feature} by Conversion Status', fontsize=12, fontweight='bold')
    ax.set_xlabel('Converted (0 = No, 1 = Yes)', fontsize=10)
    ax.set_ylabel(feature, fontsize=10)

plt.tight_layout()
plt.show()

numeric_cols = [
    'deal_value_usd', 'customer_age', 'customer_annual_income', 'customer_credit_score', 
    'customer_existing_products_count', 'customer_relationship_tenure_years', 
    'customer_debt_to_income_ratio', 'sales_rep_experience_years', 'first_contact_response_hrs',
    'days_in_pipeline', 'num_total_touchpoints', 'follow_up_consistency_score', 'converted'
]

plt.figure(figsize=(14, 10))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Heatmap of Key Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

product_conv = df.groupby('product_category')['converted'].agg(['mean', 'count']).reset_index().sort_values(by='mean', ascending=False)
sns.barplot(x='mean', y='product_category', hue='product_category', data=product_conv, palette='viridis', ax=axes[0], legend=False)
axes[0].set_title('Lead Conversion Rate by Product Category', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Average Conversion Rate', fontsize=10)
axes[0].set_ylabel('Product Category', fontsize=10)
axes[0].axvline(df['converted'].mean(), color='red', linestyle='--', linewidth=2, label=f'Global Avg ({df["converted"].mean()*100:.1f}%)')
axes[0].legend()

source_conv = df.groupby('lead_source')['converted'].agg(['mean', 'count']).reset_index().sort_values(by='mean', ascending=False)
sns.barplot(x='mean', y='lead_source', hue='lead_source', data=source_conv, palette='mako', ax=axes[1], legend=False)
axes[1].set_title('Lead Conversion Rate by Lead Source', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Average Conversion Rate', fontsize=10)
axes[1].set_ylabel('Lead Source', fontsize=10)
axes[1].axvline(df['converted'].mean(), color='red', linestyle='--', linewidth=2, label=f'Global Avg ({df["converted"].mean()*100:.1f}%)')
axes[1].legend()

plt.tight_layout()
plt.show()

df_processed = df.sample(n=min(5000, len(df)), random_state=42).copy()
print(f"Using {len(df_processed):,} records for faster model training and notebook execution.")

print("Engineering custom features...")
df_processed['income_to_deal_ratio'] = df_processed['customer_annual_income'] / (df_processed['deal_value_usd'] + 1)
df_processed['touchpoints_per_day'] = df_processed['num_total_touchpoints'] / (df_processed['days_in_pipeline'] + 1)
df_processed['credit_score_normalized'] = df_processed['customer_credit_score'] / 850  # Normalize by max credit score
df_processed['high_value_customer'] = (df_processed['customer_annual_income'] > df_processed['customer_annual_income'].median()).astype(int)

print(" Feature engineering complete!")
print(f"   New features created: income_to_deal_ratio, touchpoints_per_day, credit_score_normalized, high_value_customer")

X = df_processed.drop(columns=['lead_id', 'sales_rep_id', 'converted'])
y = df_processed['converted']


categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"  Features Prepared:")
print(f"   Numerical features ({len(numerical_cols)}): {numerical_cols}")
print(f"   Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"   Total features: {len(numerical_cols) + len(categorical_cols)}")
print(f"   Target variable distribution:\n{y.value_counts()}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Data Split Complete:")
print(f"   Train set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Test set:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"   Features per sample: {X_train.shape[1]}")
print(f"\n   Train set conversion rate: {y_train.mean()*100:.2f}%")
print(f"   Test set conversion rate:  {y_test.mean()*100:.2f}%")


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols)
    ]
)

print("  Preprocessing pipeline built successfully!")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=25, max_depth=8, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(
        n_estimators=25,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_pipelines = {}
predictions_dict = {}

print("  Training Models...\n")
for model_name, model in models.items():
    print(f"    Training {model_name}...")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    trained_pipelines[model_name] = pipeline
    
    y_pred_model = pipeline.predict(X_test)
    y_pred_proba_model = pipeline.predict_proba(X_test)[:, 1]
    predictions_dict[model_name] = {'pred': y_pred_model, 'proba': y_pred_proba_model}
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred_model),
        'Precision': precision_score(y_test, y_pred_model),
        'Recall': recall_score(y_test, y_pred_model),
        'F1-Score': f1_score(y_test, y_pred_model),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba_model)
    })
    print(f"    {model_name} completed")

df_results_numeric = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
df_results = df_results_numeric.copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
    df_results[col] = df_results[col].map(lambda x: f"{x:.4f}")

print("\n Model Performance Results:")
df_results

print("\n Model Performance Results:\n")
print(df_results.to_string(index=False))
print("\n" + "="*80)

df_results_numeric = df_results.copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
    df_results_numeric[col] = df_results_numeric[col].astype(float)

champion_name = df_results_numeric.loc[df_results_numeric['ROC-AUC'].astype(float).idxmax(), 'Model']
champion_roc_auc = df_results_numeric.loc[df_results_numeric['Model'] == champion_name, 'ROC-AUC'].values[0]

print(f"\n CHAMPION MODEL: {champion_name}")
print(f"   ROC-AUC Score: {champion_roc_auc:.4f}")
print(f"\nThis model will be used for further analysis and predictions.")

champion_pipeline = trained_pipelines[champion_name]
y_pred = predictions_dict[champion_name]['pred']
y_pred_proba = predictions_dict[champion_name]['proba']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0], 
            xticklabels=['Not Converted', 'Converted'], yticklabels=['Not Converted', 'Converted'])
axes[0].set_title(f'{champion_name}\nConfusion Matrix', fontsize=13, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color='darkorange', lw=2.5, 
             label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[1].fill_between(fpr, tpr, alpha=0.3, color='darkorange')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate', fontsize=11)
axes[1].set_ylabel('True Positive Rate', fontsize=11)
axes[1].set_title(f'{champion_name}\nROC Curve', fontsize=13, fontweight='bold')
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


print(f"\n DETAILED CLASSIFICATION REPORT - {champion_name}\n")
print(classification_report(y_test, y_pred, target_names=['Not Converted', 'Converted'], digits=4))


fig, ax = plt.subplots(figsize=(12, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

for i, model_name in enumerate(df_results_numeric['Model']):
    values = [float(df_results_numeric.loc[df_results_numeric['Model'] == model_name, m].values[0]) 
              for m in metrics]
    ax.bar(x + i*width, values, width, label=model_name, alpha=0.8)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison Across Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.05])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


print(" Extracting feature importances...")


rf_pipeline = trained_pipelines['Random Forest']
rf_model = rf_pipeline.named_steps['classifier']
preprocessor_obj = rf_pipeline.named_steps['preprocessor']

cat_features_encoded = preprocessor_obj.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
all_feature_names = numerical_cols + cat_features_encoded

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

df_importance = pd.DataFrame({
    'Feature': [all_feature_names[i] for i in indices],
    'Importance': importances[indices]
})

print(" Feature importance extraction complete!")
print(f"\nTop 20 Most Important Features:")
print(df_importance.head(20).to_string(index=False))


plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=df_importance.head(15), palette='rocket', legend=False)
plt.title('Top 15 Predictive Features Driving Lead Conversion (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=11)
plt.ylabel('Feature Name', fontsize=11)
plt.tight_layout()
plt.show()


scored_df = df_processed.copy()

all_scores = champion_pipeline.predict_proba(X)[:, 1]
all_predictions = champion_pipeline.predict(X)

scored_df['lead_score'] = all_scores
scored_df['predicted_conversion'] = all_predictions


scored_df['lead_category'] = pd.cut(
    scored_df['lead_score'],
    bins=[0, 0.3, 0.7, 1.0],
    labels=['Low Priority', 'Medium Priority', 'High Priority'],
    include_lowest=True
)

print(" Lead Scoring Complete!\n")
print("Lead Score Distribution:")
print(f"   Mean: {scored_df['lead_score'].mean():.4f}")
print(f"   Median: {scored_df['lead_score'].median():.4f}")
print(f"   Min: {scored_df['lead_score'].min():.4f}")
print(f"   Max: {scored_df['lead_score'].max():.4f}")
print("\nLead Priority Distribution:")
print(scored_df['lead_category'].value_counts().sort_index())


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(scored_df['lead_score'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(0.3, color='green', linestyle='--', linewidth=2, label='Low/Medium Threshold')
axes[0].axvline(0.7, color='red', linestyle='--', linewidth=2, label='Medium/High Threshold')
axes[0].set_title('Distribution of Lead Conversion Scores', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Lead Score', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].legend()
axes[0].grid(alpha=0.3)

priority_counts = scored_df['lead_category'].value_counts()
colors_priority = ["#99bdff", '#ffff99', "#f899ff"]
axes[1].pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%', 
            colors=colors_priority, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
axes[1].set_title('Lead Priority Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

high_priority = scored_df[scored_df['lead_category'] == 'High Priority'].sort_values('lead_score', ascending=False)

print(f"\n HIGH PRIORITY LEADS (Top 10)\n")
print(f"Total High Priority Leads: {len(high_priority):,}")
print(f"\nTop 10 Leads to Focus On:")
display_cols = ['lead_id', 'customer_age', 'customer_annual_income', 'product_category', 'lead_source', 'lead_score', 'predicted_conversion']
print(high_priority[display_cols].head(10).to_string(index=False))

sample_new_lead = X_test.iloc[[0]].copy()

sample_prediction = champion_pipeline.predict(sample_new_lead)[0]
sample_probability = champion_pipeline.predict_proba(sample_new_lead)[0][1]

print(" New Lead Prediction Completed")
print(f"Prediction: {'Likely to Convert' if sample_prediction == 1 else 'Not Likely to Convert'}")
print(f"Conversion Probability / Lead Score: {sample_probability:.4f}")

if sample_probability >= 0.70:
    priority = 'High Priority'
elif sample_probability >= 0.30:
    priority = 'Medium Priority'
else:
    priority = 'Low Priority'

print(f"Lead Priority: {priority}")
sample_new_lead

print("\n" + "="*80)
print(" STRATEGIC BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*80)

print("\n1. KEY FACTORS DRIVING CONVERSION:")
top_features = df_importance.head(5)
for idx, row in top_features.iterrows():
    print(f"   - {row['Feature']}: {row['Importance']*100:.2f}% importance")

print("\n2. CONVERSION METRICS BY CATEGORY:")
cat_analysis = df.groupby('product_category')['converted'].agg(['mean', 'count']).round(3)
for cat in cat_analysis.index:
    conv_rate = cat_analysis.loc[cat, 'mean'] * 100
    count = int(cat_analysis.loc[cat, 'count'])
    print(f"   - {cat}: {conv_rate:.1f}% conversion ({count:,} leads)")

print("\n3. LEAD SOURCE EFFECTIVENESS:")
source_analysis = df.groupby('lead_source')['converted'].agg(['mean', 'count']).sort_values('mean', ascending=False)
for source in source_analysis.index:
    conv_rate = source_analysis.loc[source, 'mean'] * 100
    count = int(source_analysis.loc[source, 'count'])
    print(f"   - {source}: {conv_rate:.1f}% conversion ({count:,} leads)")

print("\n4. PRIORITIZATION STRATEGY:")
high_count = len(scored_df[scored_df['lead_category'] == 'High Priority'])
medium_count = len(scored_df[scored_df['lead_category'] == 'Medium Priority'])
low_count = len(scored_df[scored_df['lead_category'] == 'Low Priority'])

print(f"   - High Priority Leads (Score > 0.70): {high_count:,} ({high_count/len(scored_df)*100:.1f}%)")
print(f"   - Medium Priority Leads (0.30-0.70): {medium_count:,} ({medium_count/len(scored_df)*100:.1f}%)")
print(f"   - Low Priority Leads (Score < 0.30): {low_count:,} ({low_count/len(scored_df)*100:.1f}%)")

print("\n5. MODEL PERFORMANCE:")
print(f"   - Champion Model: {champion_name}")
print(f"   - ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"   - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"   - Precision: {precision_score(y_test, y_pred):.4f}")
print(f"   - Recall: {recall_score(y_test, y_pred):.4f}")

print("\n6. ACTIONABLE RECOMMENDATIONS:")
print("   * Focus sales efforts on HIGH PRIORITY leads (>70% conversion probability)")
print("   * Optimize follow-up consistency - it's a top conversion driver")
print("   * Reduce first contact response time - faster response = higher conversion")
print("   * Prioritize high-income customers with good credit scores")
print("   * Allocate more resources to top-performing lead sources")
print("   * Use this model for real-time lead scoring to maximize ROI")

print("\n" + "="*80)

import joblib

model_path = 'champion_model.pkl'
joblib.dump(champion_pipeline, model_path)
print(f" Champion model saved: {model_path}")

results_path = 'model_results.csv'
df_results.to_csv(results_path, index=False)
print(f" Model results saved: {results_path}")

importance_path = 'feature_importance.csv'
df_importance.to_csv(importance_path, index=False)
print(f" Feature importance saved: {importance_path}")

scores_path = 'lead_scores.csv'
scored_df[['lead_id', 'lead_score', 'predicted_conversion', 'lead_category']].to_csv(scores_path, index=False)
print(f" Lead scores saved: {scores_path}")

print("\n" + "="*80)
print(" PROJECT COMPLETE!")
print("="*80)
print("\n Output Files Generated:")
print(f"   1. champion_model.pkl - Trained ML model for predictions")
print(f"   2. model_results.csv - Performance metrics of all models")
print(f"   3. feature_importance.csv - Feature importance rankings")
print(f"   4. lead_scores.csv - All leads with conversion scores")
print("\n Next Steps:")
print("   • Use lead_scores.csv to prioritize sales activities")
print("   • Deploy champion_model.pkl for real-time lead scoring")
print("   • Review feature_importance.csv for business insights")
print("   • Monitor model performance over time and retrain periodically")
print("\n" + "="*80)
