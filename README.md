# 🧠 AlphaCare-Insurance-Risk-Analytics

## 🎯 Project Objective

To uncover risk segments and build predictive models using South African car insurance data, enabling AlphaCare Insurance Solutions to:
- Optimize insurance premiums
- Identify low-risk customers for potential premium reduction
- Improve marketing strategy using data-driven insights

---

## ✅ Tasks Completed

### Task 1: Git & Exploratory Data Analysis
- Set up GitHub repository with version control and branching
- Performed detailed EDA to understand:
  - Risk patterns (loss ratios) by region, vehicle type, and customer demographics
  - Outliers and distributions in financial/vehicle features
  - Temporal trends in claims and premiums
  - Top/bottom vehicle models by claim amount
- Generated 3+ insightful visualizations

### Task 2: Data Version Control (DVC)
- Installed and configured DVC for reproducible pipelines
- Tracked raw dataset using DVC
- Integrated DVC with GitHub using `.dvc` and remote storage

### Task 3: A/B Hypothesis Testing
- Formulated and tested 4 business-critical null hypotheses:
  - Risk differences across provinces, zipcodes, gender
  - Margin differences across zipcodes
- Applied statistical tests (t-test, chi-square)
- Interpreted p-values to drive actionable business insights

### Task 4: Predictive Modeling
- Built ML models for:
  - Claim severity (`TotalClaims`) prediction
  - Premium optimization
  - Optional: Claim probability classification
- Models used: Linear Regression, Random Forest, XGBoost
- Evaluated models using RMSE, R², and classification metrics
- Applied SHAP for model interpretation and feature impact analysis

---

## 🧪 Environment Setup

```bash
# Clone the repository
git clone https://github.com/Emnet-tes/AlphaCare-Insurance-Risk-Analytics.git

# Navigate to project folder
cd AlphaCare-Insurance-Risk-Analytics

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init
dvc pull  # if needed to get the data
