# Mental Health Risk Analyzer 🧠🔍

A machine learning application that analyzes social media posts to detect signs of depression/suicide risk using Logistic Regression and XGBoost models.

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
- gdown

## Live Demo 🌐

Try the app now:  
🔗 **[https://skyler-18-mindshield-project-mental-health-analyzer-uiwplf.streamlit.app/](https://skyler-18-mindshield-project-mental-health-analyzer-uiwplf.streamlit.app/)**

## Installation ⚙️

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Skyler-18/MindShield_Project.git
   cd MindShield_Project
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
