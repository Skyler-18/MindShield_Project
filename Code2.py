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

nltk.download('stopwords')
nltk.download('wordnet')

# ======================
# 2. Data Loading & EDA
# ======================
# Load data
df = pd.read_csv('archive/Suicide_Detection.csv')
print(f"Dataset shape: {df.shape}")

# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=df)
plt.title("Class Distribution")
plt.savefig('class_dist.png')
plt.close()

# Text length analysis
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(8, 4))
sns.boxplot(x='class', y='text_length', data=df)
plt.title("Text Length by Class")
plt.savefig('text_length.png')
plt.close()

# Word clouds
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f'wordcloud_{title}.png')
    plt.close()

generate_wordcloud(" ".join(df[df['class'] == 'suicide']['text']), "Suicide Posts")
generate_wordcloud(" ".join(df[df['class'] == 'non-suicide']['text']), "Non-Suicide Posts")

# ======================
# 3. Text Preprocessing
# ======================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs, mentions
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+', '', text)
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lemmatization
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# ======================
# 4. Feature Engineering
# ======================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['cleaned_text'])
y = df['class'].map({'suicide': 1, 'non-suicide': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# 5. Model Building
# ======================
# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Logistic Regression Report:")
print(classification_report(y_test, lr_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]):.3f}")

# XGBoost
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print("\nXGBoost Report:")
print(classification_report(y_test, xgb_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]):.3f}")

# Save models
joblib.dump(lr, 'lr_model.pkl')
joblib.dump(xgb, 'xgb_model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')

# ======================
# 6. Hyperparameter Tuning
# ======================
# Logistic Regression Tuning
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression())
])

params = {
    'tfidf__max_features': [3000, 5000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'lr__C': [0.1, 1, 10]
}

lr_grid = GridSearchCV(lr_pipeline, params, cv=3, scoring='roc_auc', verbose=1)
lr_grid.fit(df['cleaned_text'], y)
print("\nBest LR Params:", lr_grid.best_params_)
print("Best LR Score:", lr_grid.best_score_)

# ======================
# 7. SHAP Analysis
# ======================
# Explain XGBoost predictions
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test[:100])  # Use subset for performance

plt.title("Feature Importance for XGBoost")
shap.summary_plot(shap_values, X_test[:100], feature_names=tfidf.get_feature_names_out(), show=False)
plt.savefig('shap_summary.png')
plt.close()

# ======================
# 8. Streamlit App
# ======================
def create_app():
    st.set_page_config(page_title="Mental Health Analyzer", layout="wide")
    
    # Load models
    tfidf = joblib.load('tfidf.pkl')
    lr = joblib.load('lr_model.pkl')
    xgb = joblib.load('xgb_model.pkl')
    
    st.title("Mental Health Risk Prediction")
    st.write("Analyze social media posts for suicide risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input Text")
        user_input = st.text_area("Paste post content here:", height=200)
        
    with col2:
        st.header("Visualizations")
        if st.button("Analyze"):
            if user_input:
                # Preprocess
                cleaned_text = clean_text(user_input)
                vectorized = tfidf.transform([cleaned_text])
                
                # Predictions
                lr_proba = lr.predict_proba(vectorized)[0][1]
                xgb_proba = xgb.predict_proba(vectorized)[0][1]
                
                # Results
                st.subheader("Results")
                st.metric("Logistic Regression", 
                         f"{'RISK DETECTED' if lr_proba > 0.5 else 'No risk'}",
                         f"{lr_proba:.1%} confidence")
                
                st.metric("XGBoost", 
                         f"{'RISK DETECTED' if xgb_proba > 0.5 else 'No risk'}",
                         f"{xgb_proba:.1%} confidence")
                
                # Visuals
                fig, ax = plt.subplots()
                ax.bar(['Logistic', 'XGBoost'], [lr_proba, xgb_proba], color=['#1f77b4', '#ff7f0e'])
                ax.set_ylim(0, 1)
                ax.set_ylabel('Risk Probability')
                st.pyplot(fig)
                
                # Show EDA
                st.subheader("Exploratory Analysis")
                cols = st.columns(3)
                with cols[0]:
                    st.image('class_dist.png', caption='Class Distribution')
                with cols[1]:
                    st.image('text_length.png', caption='Text Length Analysis')
                with cols[2]:
                    st.image('wordcloud_Suicide Posts.png', caption='Suicide Post Words')
                
                # SHAP explanation
                st.subheader("Feature Importance")
                st.image('shap_summary.png', caption='SHAP Analysis')
            else:
                st.warning("Please enter text to analyze!")

if __name__ == '__main__':
    create_app()