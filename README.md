# Mental Health Risk Analyzer 🧠🔍

A machine learning application that analyzes social media posts to detect signs of depression/suicide risk using Logistic Regression and XGBoost models.

![App Screenshot](https://via.placeholder.com/800x400?text=Mental+Health+Analyzer+Screenshot)

## Features ✨

- **Text Analysis**: Processes social media posts to detect risk indicators
- **Dual Model Prediction**: Uses both Logistic Regression and XGBoost
- **Visual Insights**: Includes word clouds and SHAP analysis
- **Interactive Interface**: Streamlit web app for easy use

## Tech Stack 🛠️

- Python 3.8+
- Scikit-learn
- XGBoost
- Streamlit
- SHAP
- NLTK

## Installation ⚙️

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mental-health-analyzer.git
   cd mental-health-analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   Download dataset from here: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
   Place it in the parent folder.

4. **Run the application**:
   First train the models:
   ```bash
   python Mental_Health_Analyzer.py
   ```

   Then launch the Streamlit app:
   ```bash
   streamlit run Mental_Health_Analyzer.py
   ```
