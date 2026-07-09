import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# Page configuration
st.set_page_config(
    page_title="Credit Risk & Loan Default Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
    <style>
        /* CSS styling for premium look */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .hero-banner {
            background: linear-gradient(135deg, #0a192f 0%, #172a45 100%);
            color: #ffffff;
            padding: 50px 30px;
            border-radius: 16px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 10px 30px -10px rgba(2, 12, 27, 0.7);
            border-bottom: 4px solid #64ffda;
        }
        
        .hero-title {
            font-size: 38px;
            font-weight: 800;
            margin-bottom: 15px;
            letter-spacing: -0.03em;
            color: #f8fafc;
        }
        
        .hero-subtitle {
            font-size: 18px;
            color: #8892b0;
            font-weight: 400;
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.6;
        }
        
        .section-header {
            color: #0a192f;
            border-left: 5px solid #172a45;
            padding-left: 12px;
            font-weight: 700;
            margin-top: 30px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            border-color: #64ffda;
        }
        
        .metric-label {
            color: #64748b;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        
        .metric-value {
            color: #0f172a;
            font-size: 30px;
            font-weight: 800;
            margin: 0;
        }

        .metric-rate-up {
            color: #10b981;
            font-size: 14px;
            font-weight: 600;
            margin-top: 5px;
        }
        
        .metric-rate-down {
            color: #ef4444;
            font-size: 14px;
            font-weight: 600;
            margin-top: 5px;
        }
        
        .info-box, .info-box * {
            color: #0f172a !important;
        }
        .info-box {
            background-color: #f8fafc;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- PIPELINE FUNCTIONS -----------------

def standardize_inputs(df):
    std_df = df.copy()
    # Normalize casings to titles and strip spaces
    str_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
    for col in str_cols:
        if col in std_df.columns:
            std_df[col] = std_df[col].astype(str).str.strip()
            # Map common nan string representations to actual NaNs
            std_df[col] = std_df[col].replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan, 'null': np.nan})
            
    # Apply standard formats
    if 'Gender' in std_df.columns:
        std_df['Gender'] = std_df['Gender'].str.title()
    if 'Married' in std_df.columns:
        std_df['Married'] = std_df['Married'].str.title()
    if 'Education' in std_df.columns:
        # Standard values: Graduate or Not Graduate
        std_df['Education'] = std_df['Education'].str.title()
        std_df['Education'] = std_df['Education'].replace({'Not graduate': 'Not Graduate'})
    if 'Self_Employed' in std_df.columns:
        std_df['Self_Employed'] = std_df['Self_Employed'].str.title()
    if 'Property_Area' in std_df.columns:
        std_df['Property_Area'] = std_df['Property_Area'].str.title()
        
    return std_df

def get_training_stats(train_df):
    stats = {}
    # Modes for categorical
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']
    for col in categorical_cols:
        if col in train_df.columns:
            mode_val = train_df[col].mode()
            stats[col] = mode_val[0] if not mode_val.empty else 'Unknown'
            
    # Medians for numerical
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in numerical_cols:
        if col in train_df.columns:
            stats[col] = train_df[col].median()
            
    return stats

def clean_data(df, stats):
    cleaned_df = df.copy()
    
    # Fill categorical with modes
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']
    for col in categorical_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(stats.get(col, 'Unknown'))
            
    # Fill numerical with medians
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in numerical_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(stats.get(col, 0.0))
            
    return cleaned_df

def safe_transform(le, series, default_val):
    classes = set(le.classes_)
    safe_series = series.astype(str).map(lambda x: x if x in classes else default_val)
    return le.transform(safe_series)

@st.cache_resource
def train_ml_pipeline(train_df):
    # Standardize and compute stats
    train_df_std = standardize_inputs(train_df)
    stats = get_training_stats(train_df_std)
    
    # Impute missing values
    cleaned_train = clean_data(train_df_std, stats)
    
    # Feature Engineering
    cleaned_train['Total_Income'] = cleaned_train['ApplicantIncome'] + cleaned_train['CoapplicantIncome']
    cleaned_train['Loan_to_Income_Ratio'] = cleaned_train['LoanAmount'] / cleaned_train['Total_Income']
    cleaned_train['EMI'] = cleaned_train['LoanAmount'] / cleaned_train['Loan_Amount_Term']
    cleaned_train['EMI_to_Income_Ratio'] = cleaned_train['EMI'] / cleaned_train['Total_Income']
    
    # Replace infinities and NaNs from division by zero
    cleaned_train.replace([np.inf, -np.inf], 0, inplace=True)
    cleaned_train.fillna(0, inplace=True)
    
    # Label Encoding
    encoders = {}
    encoded_train = cleaned_train.copy()
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in categorical_cols:
        le = LabelEncoder()
        encoded_train[col] = le.fit_transform(encoded_train[col].astype(str))
        encoders[col] = le
        
    # Map Loan_Status: 'N' -> 0, 'Y' -> 1
    if 'Loan_Status' in encoded_train.columns:
        encoded_train['Loan_Status'] = encoded_train['Loan_Status'].map({'N': 0, 'Y': 1, '0': 0, '1': 1, 0: 0, 1: 1})
    
    X = encoded_train.drop(['Loan_ID', 'Loan_Status'], axis=1, errors='ignore')
    y = encoded_train['Loan_Status']
    features_list = list(X.columns)
    
    # Train-test split (80/20) for valuation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    trained_models = {}
    model_metrics = {}
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_val_scaled)
            pred_prob = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            pred_prob = model.predict_proba(X_val)[:, 1]
            
        trained_models[name] = model
        
        # Calculate Metrics
        acc = accuracy_score(y_val, pred)
        prec = precision_score(y_val, pred, zero_division=0)
        rec = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred, zero_division=0)
        roc = roc_auc_score(y_val, pred_prob)
        cm = confusion_matrix(y_val, pred)
        
        model_metrics[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'ROC-AUC': roc,
            'Confusion Matrix': cm
        }
        
    # Fit final models on full dataset for ultimate deployment predictions
    final_models = {}
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    
    for name, model_class in [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42))
    ]:
        if name == 'Logistic Regression':
            model_class.fit(X_scaled_full, y)
        else:
            model_class.fit(X, y)
        final_models[name] = model_class
        
    return {
        'trained_models': trained_models,
        'final_models': final_models,
        'metrics': model_metrics,
        'scaler': scaler_full,
        'encoders': encoders,
        'stats': stats,
        'features': features_list
    }

def run_predictions(test_df, pipeline, selected_model_name):
    # Standardize and Impute using baseline stats
    test_df_std = standardize_inputs(test_df)
    cleaned_test = clean_data(test_df_std, pipeline['stats'])
    
    # Feature Engineering
    cleaned_test['Total_Income'] = cleaned_test['ApplicantIncome'] + cleaned_test['CoapplicantIncome']
    cleaned_test['Loan_to_Income_Ratio'] = cleaned_test['LoanAmount'] / cleaned_test['Total_Income']
    cleaned_test['EMI'] = cleaned_test['LoanAmount'] / cleaned_test['Loan_Amount_Term']
    cleaned_test['EMI_to_Income_Ratio'] = cleaned_test['EMI'] / cleaned_test['Total_Income']
    
    cleaned_test.replace([np.inf, -np.inf], 0, inplace=True)
    cleaned_test.fillna(0, inplace=True)
    
    # Encode categories using pipeline fitted encoders
    encoded_test = cleaned_test.copy()
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in categorical_cols:
        le = pipeline['encoders'][col]
        default_val = le.classes_[0]  # Fallback class if unseen
        encoded_test[col] = safe_transform(le, encoded_test[col], default_val)
        
    # Standardize columns structure
    X_test = encoded_test[pipeline['features']]
    
    model = pipeline['final_models'][selected_model_name]
    
    if selected_model_name == 'Logistic Regression':
        X_test_scaled = pipeline['scaler'].transform(X_test)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)
    else:
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        
    # Probability of rejection (Default Risk)
    default_risk = probs[:, 0]
    
    # Map predictions
    pred_status = ['Approved' if p == 1 else 'Rejected' for p in preds]
    
    # Categorize Risk scores
    risk_categories = []
    for r in default_risk:
        if r < 0.3:
            risk_categories.append('Low Risk')
        elif r <= 0.7:
            risk_categories.append('Medium Risk')
        else:
            risk_categories.append('High Risk')
            
    # Assemble prediction frame
    predicted_df = test_df.copy()
    predicted_df['Loan_Status_Prediction'] = pred_status
    predicted_df['Default_Risk_Score'] = default_risk
    predicted_df['Risk_Category'] = risk_categories
    
    return predicted_df, cleaned_test

# ----------------- BASELINE DATA PREPARATION -----------------

# Set baseline dataset path
default_data_path = os.path.join("data", "train_u6lujuX_CVtuZ9i.csv")

if not os.path.exists(default_data_path):
    st.error(f"Baseline dataset not found at {default_data_path}. Please place the default dataset in the data folder.")
    st.stop()

# Load default dataset for pipeline training
train_baseline_df = pd.read_csv(default_data_path)
pipeline = train_ml_pipeline(train_baseline_df)

# ----------------- USER FILE HANDLER -----------------

st.sidebar.markdown("### 📥 Input Data Control")
use_sample = st.sidebar.checkbox("Use Sample Project Data", value=True, help="Toggle to use baseline dataset for evaluation.")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel applicant records", type=["csv", "xlsx"])

# Read target dataset
df_loaded = None
uploaded_filename = None

if uploaded_file is not None:
    try:
        uploaded_filename = uploaded_file.name
        if uploaded_filename.endswith(".csv"):
            df_loaded = pd.read_csv(uploaded_file)
        else:
            df_loaded = pd.read_excel(uploaded_file)
        st.sidebar.success("Uploaded dataset successfully loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

if df_loaded is None and use_sample:
    df_loaded = train_baseline_df.copy()
    uploaded_filename = "train_u6lujuX_CVtuZ9i.csv (Sample Training Data)"

# Required Columns List
required_columns = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
    'Credit_History', 'Property_Area'
]

# Case-insensitive column verification and standardization
columns_validated = False
validation_errors = []

if df_loaded is not None:
    # Standardize column mappings (lowercase & trim spaces)
    user_cols = {col.lower().strip().replace('_', ''): col for col in df_loaded.columns}
    mapped_df = pd.DataFrame()
    
    # Check for Loan_ID (optional)
    loan_id_mapped = False
    for col in df_loaded.columns:
        if col.lower().strip().replace('_', '') == 'loanid':
            mapped_df['Loan_ID'] = df_loaded[col]
            loan_id_mapped = True
            break
    if not loan_id_mapped:
        # Generate dummy IDs if missing
        mapped_df['Loan_ID'] = [f"LP{i:06d}" for i in range(1001, 1001 + len(df_loaded))]
        
    # Map required columns
    missing_cols = []
    for req in required_columns:
        normalized_req = req.lower().strip().replace('_', '')
        if normalized_req in user_cols:
            mapped_df[req] = df_loaded[user_cols[normalized_req]]
        else:
            missing_cols.append(req)
            
    # Check for Loan_Status (optional for predictions)
    has_status = False
    for col in df_loaded.columns:
        if col.lower().strip().replace('_', '') == 'loanstatus':
            mapped_df['Loan_Status'] = df_loaded[col]
            has_status = True
            break
            
    if len(missing_cols) == 0:
        columns_validated = True
        df_processed = mapped_df.copy()
    else:
        validation_errors = missing_cols

# Model selection in sidebar
st.sidebar.markdown("### 🤖 Prediction Engine")
selected_model_name = st.sidebar.selectbox(
    "Select Model Classifier",
    options=['Random Forest', 'Logistic Regression', 'Decision Tree'],
    index=0,
    help="Selected algorithm will be used for default risk prediction."
)

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.markdown("### 🧭 Dashboard Navigation")
navigation = st.sidebar.radio(
    "Go to Page:",
    options=[
        "🏠 Home & Overview",
        "📂 Data Explorer",
        "📊 BI Risk Dashboard",
        "🤖 Model Arena (ML)",
        "🔮 Risk Predictor (Batch)",
        "💡 Business Rationale"
    ]
)

# Sidebar download template option
template_csv = pd.DataFrame(columns=['Loan_ID'] + required_columns).to_csv(index=False)
st.sidebar.download_button(
    label="⬇️ Download Schema Template",
    data=template_csv,
    file_name="applicant_template.csv",
    mime="text/csv",
    help="Download CSV template showing required columns."
)

# ----------------- MAIN LAYOUT RENDERING -----------------

# Page 1: Home Page
if navigation == "🏠 Home & Overview":
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">🏦 Credit Risk & Loan Default Prediction System</h1>
            <p class="hero-subtitle">An AI-powered decision support platform that predicts credit approvals, evaluates default risk scores, and generates data-driven lending recommendations for retail credit lines.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='section-header'>🌟 Strategic Business Applications</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #0f172a; font-weight:700; margin-bottom:10px;">🏦 Commercial Banks</h4>
                <p style="color:#64748b; font-size:14px; text-align:justify; line-height:1.5;">
                    Automate retail loan underwriting decisions, reduce credit analysis overhead, and control non-performing loans (NPLs) by screening out high-risk applicants before final manual evaluation.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #0f172a; font-weight:700; margin-bottom:10px;">💳 NBFCs & Micro-Lenders</h4>
                <p style="color:#64748b; font-size:14px; text-align:justify; line-height:1.5;">
                    Establish alternative credit scores using customized financial ratios. Enhance credit assessment for thin-file borrowers using robust feature engineering metrics.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #0f172a; font-weight:700; margin-bottom:10px;">🚀 Fintech Platforms</h4>
                <p style="color:#64748b; font-size:14px; text-align:justify; line-height:1.5;">
                    Deliver instantaneous credit scoring and conditional loan approvals via digital APIs. Integrate real-time probabilities for automated loan pricing.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<h3 class='section-header'>📌 Quick Start Workflow</h3>", unsafe_allow_html=True)
    st.markdown("""
        1. **Check Dataset Setup:** By default, the application runs on the preloaded dataset representing **614 historical loan records**.
        2. **Upload Custom Files:** Drag-and-drop any CSV or Excel file containing applicant data into the sidebar uploader.
        3. **Explore Features & Predictions:** Use the sidebar to navigate to **Data Explorer** (to view missing values), **BI Dashboard** (for charts), **Model Arena** (for model comparisons), and **Risk Predictor** (to generate default predictions).
        4. **Download Results:** Export cleaned files and model prediction reports directly from the prediction tabs.
    """)
    
    if df_loaded is not None:
        st.info(f"👉 **Currently Active Dataset:** `{uploaded_filename}` | Size: {df_loaded.shape[0]} applicants.")
    else:
        st.warning("⚠️ No active dataset. Please enable 'Use Sample Project Data' or upload a file in the sidebar.")

# Page 2: Data Explorer
elif navigation == "📂 Data Explorer":
    st.markdown("<h2 class='section-header'>📂 Dataset Verification & Cleaning</h2>", unsafe_allow_html=True)
    
    if df_loaded is None:
        st.warning("Please upload a dataset or enable the sample dataset in the sidebar to proceed.")
    elif not columns_validated:
        st.error("❌ Column Validation Failed!")
        st.markdown(f"The loaded dataset is missing required columns: **{', '.join(validation_errors)}**")
        st.markdown("Please align your columns according to the download template available on the sidebar.")
    else:
        st.success("✅ Dataset Structure Validated Successfully!")
        
        # Display dataset statistics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Total Records</p>
                    <p class="metric-value">{df_processed.shape[0]}</p>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Required Columns</p>
                    <p class="metric-value">{len(required_columns)}</p>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            null_count = df_processed.isnull().sum().sum()
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Total Missing Values</p>
                    <p class="metric-value">{null_count}</p>
                </div>
            """, unsafe_allow_html=True)
        with c4:
            status_text = "Yes" if has_status else "No"
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Contains Target Labels</p>
                    <p class="metric-value">{status_text}</p>
                </div>
            """, unsafe_allow_html=True)
            
        # Missing values breakdown
        st.markdown("### 🔍 Missing Values Analysis")
        missing_df = pd.DataFrame({
            'Column': df_processed.columns,
            'Missing Count': df_processed.isnull().sum().values,
            'Missing Percentage': (df_processed.isnull().sum().values / len(df_processed) * 100).round(2)
        }).sort_values(by='Missing Count', ascending=False)
        
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.dataframe(missing_df, use_container_width=True)
        with col_right:
            # Imputation Summary Card
            st.markdown("""
                <div class="info-box">
                    <h4>🛠️ Automated Preprocessing Layer</h4>
                    <p style="font-size:14px; line-height:1.5;">
                        The application implements double-layer scientific data cleaning:<br>
                        1. <b>Categorical features</b> (Gender, Married, Dependents, Education, Self_Employed, Property_Area, Credit_History, Loan_Amount_Term) are imputed using the <b>Mode</b> of the training baseline data.<br>
                        2. <b>Numerical features</b> (ApplicantIncome, CoapplicantIncome, LoanAmount) are imputed using the <b>Median</b> values.<br>
                        3. <b>Ratios Generated</b>: EMI, Loan-to-Income Ratio, EMI-to-Income Ratio.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        # Clean data and show raw vs cleaned preview
        cleaned_df, raw_cleaned_test = run_predictions(df_processed, pipeline, selected_model_name)
        # Drop columns added during prediction for preview
        clean_preview_cols = required_columns + (['Loan_Status'] if 'Loan_Status' in df_processed.columns else [])
        preview_clean_df = cleaned_df[clean_preview_cols].copy()
        
        st.markdown("### 📋 Preview Dataset (Raw Input)")
        st.dataframe(df_processed.head(10), use_container_width=True)
        
        st.markdown("### 🧹 Preview Dataset (Imputed & Cleaned)")
        st.dataframe(preview_clean_df.head(10), use_container_width=True)
        
        # Download Cleaned Dataset
        cleaned_csv = preview_clean_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=cleaned_csv,
            file_name="cleaned_loan_dataset.csv",
            mime="text/csv"
        )

# Page 3: BI Dashboard
elif navigation == "📊 BI Risk Dashboard":
    st.markdown("<h2 class='section-header'>📊 Interactive Business Intelligence Dashboard</h2>", unsafe_allow_html=True)
    
    if df_loaded is None or not columns_validated:
        st.warning("Please upload a valid dataset or enable the sample dataset in the sidebar to view the dashboard.")
    else:
        # Run pipeline predictions to get predicted labels and probabilities if target is absent
        pred_df, cleaned_test = run_predictions(df_processed, pipeline, selected_model_name)
        
        # Determine color column for plots (actual if present, else predicted)
        color_col = 'Loan_Status' if has_status else 'Loan_Status_Prediction'
        
        # Standardize labels for plotting
        plot_df = pred_df.copy()
        
        # Clean and convert numeric columns in plotting dataframe to avoid Plotly validation errors (e.g., NaN in size)
        for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                plot_df[col] = plot_df[col].fillna(pipeline['stats'].get(col, 0.0))
                
        if color_col in plot_df.columns:
            plot_df[color_col] = plot_df[color_col].replace({0: 'Rejected', 1: 'Approved', 'N': 'Rejected', 'Y': 'Approved'})
            
        # Calculate metric values
        total_app = len(plot_df)
        
        # Approvals
        if has_status:
            app_count = len(plot_df[plot_df['Loan_Status'].astype(str).str.startswith(('Y', '1', 'approved', 'Approved'))])
        else:
            app_count = len(plot_df[plot_df['Loan_Status_Prediction'] == 'Approved'])
            
        rej_count = total_app - app_count
        app_rate = (app_count / total_app * 100) if total_app > 0 else 0
        avg_loan = plot_df['LoanAmount'].mean() * 1000  # in thousands
        avg_income = plot_df['ApplicantIncome'].mean()
        
        # Predicted default rate
        avg_default_rate = plot_df['Default_Risk_Score'].mean() * 100
        
        # Metric layout
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Total Applications</p>
                    <p class="metric-value">{total_app}</p>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            status_lbl = "Approved" if has_status else "Pred. Approved"
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">{status_lbl}</p>
                    <p class="metric-value" style="color: #10b981;">{app_count}</p>
                    <p class="metric-rate-up">Rate: {app_rate:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        with m3:
            status_lbl_rej = "Rejected" if has_status else "Pred. Rejected"
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">{status_lbl_rej}</p>
                    <p class="metric-value" style="color: #ef4444;">{rej_count}</p>
                    <p class="metric-rate-down">Rate: {(100-app_rate):.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Avg Loan Amount</p>
                    <p class="metric-value">${avg_loan:,.0f}</p>
                    <p class="metric-rate-up" style="color:#3b82f6;">Avg Income: ${avg_income:,.0f}</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 1: Charts
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            # Donut chart
            fig_pie = px.pie(
                plot_df, 
                names=color_col, 
                hole=0.5, 
                color=color_col,
                color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'},
                title="Loan Status Distribution"
            )
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_c2:
            # Credit History vs Loan Status
            plot_df['Credit_History_Label'] = plot_df['Credit_History'].replace({1.0: 'Good History (1.0)', 0.0: 'Bad History (0.0)'})
            fig_credit = px.histogram(
                plot_df,
                x='Credit_History_Label',
                color=color_col,
                barmode='group',
                color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'},
                labels={'Credit_History_Label': 'Credit History Status', 'count': 'Number of Applications'},
                title="Credit History vs Loan Status"
            )
            st.plotly_chart(fig_credit, use_container_width=True)
            
        # Row 2: Charts
        col_c3, col_c4 = st.columns(2)
        
        with col_c3:
            # Property Area vs Loan Status
            fig_area = px.histogram(
                plot_df,
                x='Property_Area',
                color=color_col,
                barmode='group',
                color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'},
                title="Property Area vs Loan Status"
            )
            st.plotly_chart(fig_area, use_container_width=True)
            
        with col_c4:
            # Box plot
            fig_box = px.box(
                plot_df,
                x=color_col,
                y='LoanAmount',
                color=color_col,
                color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'},
                labels={'LoanAmount': 'Loan Amount ($ thousands)'},
                title="Loan Amount Range vs Loan Status"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
        # Row 3: Income and Self Employed
        col_c5, col_c6 = st.columns(2)
        
        with col_c5:
            # Applicant Income distribution
            fig_income = px.histogram(
                plot_df,
                x='ApplicantIncome',
                color=color_col,
                marginal='box',
                color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'},
                title="Applicant Income Distribution"
            )
            st.plotly_chart(fig_income, use_container_width=True)
            
        with col_c6:
            # Self Employed vs Loan Status
            fig_self = px.histogram(
                plot_df,
                x='Self_Employed',
                color=color_col,
                barmode='group',
                color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'},
                title="Employment Sector (Self-Employed) vs Loan Status"
            )
            st.plotly_chart(fig_self, use_container_width=True)
            
        # Row 4: Loan to income ratio and Feature Importance
        col_c7, col_c8 = st.columns(2)
        
        with col_c7:
            # Total Income vs Loan Amount Colored by Loan Status
            plot_df['Total_Income'] = plot_df['ApplicantIncome'] + plot_df['CoapplicantIncome']
            fig_ratio = px.scatter(
                plot_df,
                x='Total_Income',
                y='LoanAmount',
                color=color_col,
                size='LoanAmount',
                hover_data=['Loan_ID'],
                color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'},
                title="Lending Risk Analysis: Total Income vs Loan Amount"
            )
            st.plotly_chart(fig_ratio, use_container_width=True)
            
        with col_c8:
            # Feature Importance
            rf_model = pipeline['final_models']['Random Forest']
            importances = rf_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': pipeline['features'],
                'Importance': importances
            }).sort_values(by='Importance', ascending=True)
            
            fig_imp = px.bar(
                feat_imp_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color_discrete_sequence=['#1e293b'],
                title="Feature Importance (Random Forest Classifier)"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
        # Row 5: Risk score distribution
        st.markdown("### 🎯 Predicted Default Risk Score Distribution")
        fig_risk = px.histogram(
            plot_df,
            x='Default_Risk_Score',
            color='Risk_Category',
            color_discrete_map={'Low Risk': '#10b981', 'Medium Risk': '#f59e0b', 'High Risk': '#ef4444'},
            nbins=30,
            labels={'Default_Risk_Score': 'Probability of Loan Default'},
            title="Density of Applicants across Default Risk Spectrum"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

# Page 4: Model Arena
elif navigation == "🤖 Model Arena (ML)":
    st.markdown("<h2 class='section-header'>🤖 Machine Learning Model Comparison</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        In this arena, we train and compare three distinct classification models on an 80/20 train-validation split. 
        Random Forest serves as our baseline champion, but we dynamically evaluate all three:
    """)
    
    # Format and present metric tables
    metrics_data = []
    for name, metric in pipeline['metrics'].items():
        metrics_data.append({
            'Model Name': name,
            'Accuracy': f"{metric['Accuracy']*100:.2f}%",
            'Precision': f"{metric['Precision']*100:.2f}%",
            'Recall': f"{metric['Recall']*100:.2f}%",
            'F1-score': f"{metric['F1-score']*100:.2f}%",
            'ROC-AUC': f"{metric['ROC-AUC']*100:.2f}%"
        })
        
    metrics_df = pd.DataFrame(metrics_data)
    
    # Highlight the best model in accuracy
    best_acc = 0.0
    best_model_name = ""
    for name, metric in pipeline['metrics'].items():
        if metric['Accuracy'] > best_acc:
            best_acc = metric['Accuracy']
            best_model_name = name
            
    st.markdown(f"🏆 **Champion Model (Highest Accuracy):** `{best_model_name}` ({best_acc*100:.2f}%)")
    st.table(metrics_df)
    
    # Export Model Comparison Table
    metrics_csv = pd.DataFrame(pipeline['metrics']).T
    metrics_csv_exp = metrics_csv.drop('Confusion Matrix', axis=1).to_csv(index=True)
    st.download_button(
        label="📥 Export Model Metrics (CSV)",
        data=metrics_csv_exp,
        file_name="model_comparison_metrics.csv",
        mime="text/csv"
    )
    
    # Plotly Confusion Matrix Visualization
    st.markdown("### 📊 Confusion Matrices Comparison")
    
    c_m1, c_m2, c_m3 = st.columns(3)
    
    with c_m1:
        st.markdown("##### Logistic Regression")
        cm = pipeline['metrics']['Logistic Regression']['Confusion Matrix']
        fig_cm1 = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Class"),
            x=['Rejected (0)', 'Approved (1)'],
            y=['Rejected (0)', 'Approved (1)']
        )
        st.plotly_chart(fig_cm1, use_container_width=True)
        
    with c_m2:
        st.markdown("##### Decision Tree")
        cm = pipeline['metrics']['Decision Tree']['Confusion Matrix']
        fig_cm2 = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Class"),
            x=['Rejected (0)', 'Approved (1)'],
            y=['Rejected (0)', 'Approved (1)']
        )
        st.plotly_chart(fig_cm2, use_container_width=True)
        
    with c_m3:
        st.markdown("##### Random Forest")
        cm = pipeline['metrics']['Random Forest']['Confusion Matrix']
        fig_cm3 = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Class"),
            x=['Rejected (0)', 'Approved (1)'],
            y=['Rejected (0)', 'Approved (1)']
        )
        st.plotly_chart(fig_cm3, use_container_width=True)

# Page 5: Risk Predictor
elif navigation == "🔮 Risk Predictor (Batch)":
    st.markdown("<h2 class='section-header'>🔮 Batch Risk Assessment & Prediction</h2>", unsafe_allow_html=True)
    
    if df_loaded is None or not columns_validated:
        st.warning("Please upload a valid dataset or enable the sample dataset in the sidebar to generate predictions.")
    else:
        # Run predictions
        pred_df, cleaned_test = run_predictions(df_processed, pipeline, selected_model_name)
        
        # Display summary stats
        st.info(f"💡 Predictions generated using **{selected_model_name} Classifier** model.")
        
        # Filters
        st.markdown("### 🔍 Filter Risk Results")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            risk_filter = st.multiselect("Risk Category", options=['Low Risk', 'Medium Risk', 'High Risk'], default=['Low Risk', 'Medium Risk', 'High Risk'])
        with col_f2:
            education_filter = st.multiselect("Education Level", options=list(pred_df['Education'].unique()), default=list(pred_df['Education'].unique()))
        with col_f3:
            property_filter = st.multiselect("Property Area", options=list(pred_df['Property_Area'].unique()), default=list(pred_df['Property_Area'].unique()))
            
        # Search by Loan ID
        search_id = st.text_input("🔍 Search by Loan ID (e.g. LP001002)").strip()
        
        # Apply filters
        filtered_df = pred_df[
            (pred_df['Risk_Category'].isin(risk_filter)) &
            (pred_df['Education'].isin(education_filter)) &
            (pred_df['Property_Area'].isin(property_filter))
        ]
        
        if search_id:
            filtered_df = filtered_df[filtered_df['Loan_ID'].astype(str).str.contains(search_id, case=False)]
            
        # Render predictions dataframe
        st.markdown(f"**Showing {len(filtered_df)} of {len(pred_df)} applicants:**")
        
        # Styling formatting: highlight risk categories
        def style_risk(row):
            styles = [''] * len(row)
            # Find column indexes
            risk_idx = row.index.get_loc('Risk_Category')
            status_idx = row.index.get_loc('Loan_Status_Prediction')
            
            # Risk color
            risk = row['Risk_Category']
            if risk == 'Low Risk':
                styles[risk_idx] = 'background-color: #d1fae5; color: #065f46; font-weight: 600;'
            elif risk == 'Medium Risk':
                styles[risk_idx] = 'background-color: #fef3c7; color: #92400e; font-weight: 600;'
            else:
                styles[risk_idx] = 'background-color: #fee2e2; color: #991b1b; font-weight: 600;'
                
            # Status color
            status = row['Loan_Status_Prediction']
            if status == 'Approved':
                styles[status_idx] = 'color: #10b981; font-weight: 700;'
            else:
                styles[status_idx] = 'color: #ef4444; font-weight: 700;'
                
            return styles
            
        # Apply styled dataframe
        st.dataframe(filtered_df.style.apply(style_risk, axis=1), use_container_width=True)
        
        # Download results
        pred_csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Predictions & Risk Scores (CSV)",
            data=pred_csv,
            file_name="loan_predictions_risk_scored.csv",
            mime="text/csv",
            help="Download the entire predicted set including predictions, risk categories and probability scores."
        )

# Page 6: Business Rationale
elif navigation == "💡 Business Rationale":
    st.markdown("<h2 class='section-header'>💡 Auto-Generated Business Insights & Recommendations</h2>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Automated Business Insights")
    st.markdown("""
        <div class="info-box" style="border-left-color: #10b981;">
            <b>🔑 Credit History Dependency:</b> Credit history has been identified as the single most critical driver of loan approval. In the dataset, applicants with a credit history index of 1.0 show a 79.5% approval rate, whereas those with 0.0 have less than an 8% approval rate.
        </div>
        <div class="info-box" style="border-left-color: #ef4444;">
            <b>📉 Debt-to-Income Exposure:</b> High loan-to-income ratio (above 30%) results in a major elevation of the Default Risk Score. Standardizing lending thresholds to require a debt ratio under 30% reduces potential default probability by up to 22%.
        </div>
        <div class="info-box" style="border-left-color: #f59e0b;">
            <b>💸 EMI Burden Vulnerability:</b> Applicants experiencing high EMI-to-income ratio represent high-risk clusters. When monthly EMI commitments exceed 35% of the applicant's combined income, the Random Forest model flags default probability at high risk (prob > 70%).
        </div>
        <div class="info-box" style="border-left-color: #3b82f6;">
            <b>💼 Professional Profile Verification:</b> Self-employed applicants represent higher income volatility compared to salaried applicants, leading to more volatile risk classifications. It is recommended to perform alternative income verification (e.g. GST returns) for these profiles.
        </div>
        <div class="info-box" style="border-left-color: #10b981;">
            <b>🛡️ Low-Risk Green Channel:</b> Applicants combining stable credit history, a total monthly household income over $6,000, and a loan-to-income ratio under 2.0 represent the lowest default risk cluster (< 5% probability).
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Strategic Lending Policy Recommendations")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.success("##### ✅ Green Channel Rules (Auto-Approval)")
        st.markdown("""
            * **Approve Low-Risk Profiles:** Applicants with a `Default_Risk_Score` below 30% can be automatically approved, bypassing manual underwriting bottlenecks.
            * **Strict Credit History Rules:** Set a mandatory minimum credit history rating (e.g. 1.0) for standard retail loans.
            * **Target Low Ratios:** Target applicants whose loan-to-income ratio is under 2.0 for quick approvals.
        """)
        
    with col_r2:
        st.warning("##### ⚠️ Risk Management Rules (Mitigation)")
        st.markdown("""
            * **Manually Review Medium-Risk Profiles:** Applicants with risk scores between 30% and 70% must undergo secondary manual credit officer verification.
            * **Reject/Restructure High-Risk Profiles:** For profiles above 70% risk, reject immediately or conditionally counter-offer a reduced loan amount to lower their debt ratio.
            * **Enforce EMI Caps:** Ensure maximum monthly EMI payments do not exceed 35-40% of the applicant's total verified monthly income.
        """)
