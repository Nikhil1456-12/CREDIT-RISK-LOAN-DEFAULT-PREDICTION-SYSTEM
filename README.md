# Credit Risk & Loan Default Prediction System

A complete retail credit line underwriting decision support system built in Python using **Streamlit**, **scikit-learn**, and **Plotly**. This application cleans loan applicant data, performs feature engineering, evaluates models (Logistic Regression, Decision Trees, and Random Forests), generates probability scores (Default Risk), and categorizes borrowers (Low, Medium, High Risk) with auto-generated business policy recommendations.

---

## 🚀 Core Features

1. **🏠 Home & Overview:** Strategic business landing page outlining fintech/banking use cases and a quick uploader supporting both `.csv` and `.xlsx` files.
2. **📂 Data Explorer:** Automatically imputes missing values (median for numerical, mode for categorical), runs feature engineering (EMI, Debt-to-Income, etc.), and validates file columns.
3. **📊 BI Dashboard:** Fully interactive graphs and charts showing Credit History, Property Area, Loan Amount Box plots, Income densities, and Risk Score distribution.
4. **🤖 Model Arena:** Compares Logistic Regression, Decision Tree, and Random Forest models on Accuracy, Precision, Recall, F1, and ROC-AUC metrics with corresponding confusion matrices.
5. **🔮 Batch Risk Predictor:** Runs risk scoring and categorization on uploaded applicant lists, with interactive sidebar filters, search by Loan ID, and one-click predictions CSV export.
6. **💡 Business Hub:** Renders auto-generated policy rules (Green Channel rules, Risk Mitigation Capping) based on the model's feature relationships.

---

## 📂 Expected Dataset Schema

The system supports CSV or Excel uploads with the following columns:

| Column Name | Description | Example Values |
| :--- | :--- | :--- |
| **Loan_ID** *(Optional)* | Unique applicant identifier | LP001002, LP001003 |
| **Gender** | Applicant gender | Male, Female |
| **Married** | Marital status | Yes, No |
| **Dependents** | Number of dependents | 0, 1, 2, 3+ |
| **Education** | Level of education | Graduate, Not Graduate |
| **Self_Employed** | Employment category | Yes, No |
| **ApplicantIncome** | Monthly income of primary applicant | 5849, 4583 |
| **CoapplicantIncome** | Monthly income of co-applicant | 0, 1508 |
| **LoanAmount** | Total loan amount in thousands | 128, 66 |
| **Loan_Amount_Term** | Term of loan in months | 360, 180 |
| **Credit_History** | Baseline credit rating index | 1.0 (Good), 0.0 (Bad) |
| **Property_Area** | Urban development category | Urban, Semiurban, Rural |
| **Loan_Status** *(Optional)* | Historical approval status | Y (Yes), N (No) |

*Note: If `Loan_Status` is present in the uploaded file, the app runs in **Evaluation Mode** (evaluating metrics). If it is missing, the app runs in **Prediction Mode** (generating classifications and default risk scores).*

---

## 💻 Local Installation & Setup

1. **Ensure Python 3.9+ is installed.**
2. **Clone/Navigate to the project directory:**
   ```bash
   cd Credit-Risk-Loan-Default-Prediction
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
5. Open your web browser and navigate to `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Cloud

To share the application as an interactive public dashboard:

1. **Push your code to a GitHub Repository** (containing `app.py`, `requirements.txt`, and the `data/` folder).
2. **Log into [Streamlit Cloud](https://share.streamlit.io/).**
3. Click **New app**, select your GitHub repository, branch, and set the Main file path to `app.py`.
4. Click **Deploy!** Streamlit Cloud will automatically build the environment and host your app.

---

## 📸 Screenshots

*(Placeholder for Application Dashboard mockups)*
- *Landing Page & File Uploader*
- *Interactive BI Visual Dashboard*
- *Machine Learning Model Arena*
- *Batch Risk Predictor & Score Filter*
