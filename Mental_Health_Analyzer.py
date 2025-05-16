# ======================
# 1. Setup & Imports
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import shap
import joblib
import streamlit as st
from wordcloud import WordCloud
import os

nltk.download('stopwords')
nltk.download('wordnet')

# Check for existing artifacts
MODEL_FILES_EXIST = all([
    os.path.exists('lr_model.pkl'),
    os.path.exists('xgb_model.pkl'),
    os.path.exists('tfidf.pkl'),
    os.path.exists('class_dist.png'),
    os.path.exists('shap_summary.png')
])

# ======================
# 2. Training Function
# ======================
def train_models():
    """Run all training and setup steps"""
    print("Starting model training...")
    
    # Load data
    df = pd.read_csv('Suicide_Detection.csv')
    print(f"Dataset shape: {df.shape}")

    # EDA Visualizations
    plt.figure(figsize=(6, 4))
    sns.countplot(x='class', data=df)
    plt.title("Class Distribution")
    plt.savefig('class_dist.png')
    plt.close()

    df['text_length'] = df['text'].apply(len)
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='class', y='text_length', data=df)
    plt.title("Text Length by Class")
    plt.savefig('text_length.png')
    plt.close()

    def generate_wordcloud(text, title):
        wordcloud = WordCloud(width=800, height=400).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')
        plt.savefig(f'wordcloud_{title}.png')
        plt.close()

    generate_wordcloud(" ".join(df[df['class'] == 'suicide']['text']), "Depressed Posts")
    generate_wordcloud(" ".join(df[df['class'] == 'non-suicide']['text']), "Non-Depressed Posts")

    # Text Preprocessing
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+|@\w+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text

    df['cleaned_text'] = df['text'].apply(clean_text)

    # Feature Engineering
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['class'].map({'suicide': 1, 'non-suicide': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Building
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    print("Logistic Regression trained!")
    
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    xgb.fit(X_train, y_train)
    print("XGBoost trained!")

    # Save models
    joblib.dump(lr, 'lr_model.pkl')
    joblib.dump(xgb, 'xgb_model.pkl')
    joblib.dump(tfidf, 'tfidf.pkl')

    # SHAP Analysis
    print("\nGenerating SHAP analysis...")
    explainer = shap.Explainer(xgb)
    shap_values = explainer(X_test[:100])
    plt.title("Feature Importance for XGBoost")
    shap.summary_plot(shap_values, X_test[:100], feature_names=tfidf.get_feature_names_out(), show=False)
    plt.savefig('shap_summary.png')
    plt.close()
    
    print("\nAll models and artifacts saved successfully!")

# ======================
# 3. Streamlit App
# ======================
def create_app():
    st.set_page_config(page_title="Mental Health Analyzer", layout="wide")
    
    if not MODEL_FILES_EXIST:
        st.warning("Required model files not found. Training models...")
        with st.spinner("Training models (this may take a few minutes)"):
            train_models()
        st.success("Training complete! Please refresh the page.")
        return
    
    # Load models
    try:
        tfidf = joblib.load('tfidf.pkl')
        lr = joblib.load('lr_model.pkl')
        xgb = joblib.load('xgb_model.pkl')
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    st.title("Mental Health Risk Prediction")
    st.write("Analyze social media posts for depression/suicide risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input Text")
        user_input = st.text_area("Paste post content here:", height=200, key="input_text")
        
    with col2:
        st.header("Analysis Results")
        if st.button("Analyze", key="analyze_btn"):
            if user_input:
                # Preprocess
                cleaned_text = clean_text(user_input)
                vectorized = tfidf.transform([cleaned_text])
                
                # Predictions
                lr_proba = lr.predict_proba(vectorized)[0][1]
                xgb_proba = xgb.predict_proba(vectorized)[0][1]
                
                # Results
                st.subheader("Risk Assessment")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Logistic Regression", 
                            "High Risk" if lr_proba > 0.5 else "Low Risk",
                            f"{lr_proba:.1%} confidence")
                with col_b:
                    st.metric("XGBoost", 
                            "High Risk" if xgb_proba > 0.5 else "Low Risk",
                            f"{xgb_proba:.1%} confidence")
                
                # Probability visualization
                st.subheader("Model Confidence")
                fig, ax = plt.subplots()
                ax.bar(['Logistic', 'XGBoost'], [lr_proba, xgb_proba], color=['#1f77b4', '#ff7f0e'])
                ax.set_ylim(0, 1)
                ax.set_ylabel('Risk Probability')
                st.pyplot(fig)
                
                # EDA Visualizations
                st.subheader("Data Insights")
                tab1, tab2, tab3 = st.tabs(["Class Distribution", "Text Analysis", "Important Features"])
                
                with tab1:
                    st.image('class_dist.png')
                    st.image('wordcloud_Depression Posts.png')
                    
                with tab2:
                    st.image('text_length.png')
                    st.image('wordcloud_Non-Depression Posts.png')
                    
                with tab3:
                    st.image('shap_summary.png')
            else:
                st.warning("Please enter text to analyze!")

# Text cleaning function (needed for app)
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# ======================
# 4. Main Execution
# ======================
if __name__ == '__main__':
    if not MODEL_FILES_EXIST and not st.runtime.exists():
        print("Training models first...")
        train_models()
        print("Now run: streamlit run Code2.py")
    else:
        create_app()