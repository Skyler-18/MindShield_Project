# ======================
# 1. Setup & Data Loading
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('archive/Suicide_Detection.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

# ======================
# 2. Exploratory Data Analysis (EDA)
# ======================
# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=df)
plt.title("Class Distribution (Suicide vs Non-Suicide)")
plt.show()

# Text length analysis
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(8, 4))
sns.boxplot(x='class', y='text_length', data=df)
plt.title("Text Length Distribution by Class")
plt.show()

# Word cloud for suicide class
from wordcloud import WordCloud
suicide_text = " ".join(df[df['class'] == 'suicide']['text'])
wordcloud = WordCloud(width=800, height=400).generate(suicide_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Frequent Words in Suicide Posts")
plt.axis('off')
plt.show()

# ======================
# 3. Text Preprocessing
# ======================
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(clean_text)
df['lemmatized_text'] = df['cleaned_text'].apply(lemmatize_text)

# ======================
# 4. Feature Engineering
# ======================
# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words=stopwords.words('english'),
    ngram_range=(1, 2)  # Include bigrams
)
X = tfidf.fit_transform(df['lemmatized_text'])
y = df['class'].map({'suicide': 1, 'non-suicide': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 5. Model Building
# ======================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear', probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# ======================
# 6. Hyperparameter Tuning (Best Model)
# ======================
# Pipeline for Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('clf', LogisticRegression())
])

params = {
    'tfidf__max_features': [3000, 5000, 7000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(
    pipeline, params, cv=3, scoring='roc_auc', verbose=1
)
grid_search.fit(df['lemmatized_text'], y)

print("\nBest Parameters:", grid_search.best_params_)
print("Best ROC-AUC:", grid_search.best_score_)

# ======================
# 7. Interpretability (SHAP)
# ======================
import shap

# Train best model
best_model = grid_search.best_estimator_
best_model.fit(df['lemmatized_text'], y)

# SHAP analysis
explainer = shap.Explainer(best_model.named_steps['clf'], 
                          masker=best_model.named_steps['tfidf'].transform(df['lemmatized_text'][:100]))
shap_values = explainer(best_model.named_steps['tfidf'].transform(df['lemmatized_text'][:100]))

# Plot top features
shap.plots.beeswarm(shap_values)
plt.title("Top Features Influencing Suicide Risk Prediction")
plt.show()

# ======================
# 8. Deployment-Ready Function
# ======================
def predict_mental_risk(text):
    cleaned = clean_text(text)
    lemmatized = lemmatize_text(cleaned)
    proba = best_model.predict_proba([lemmatized])[0][1]
    return {
        "risk_probability": float(proba),
        "prediction": "suicide" if proba > 0.5 else "non-suicide"
    }

# Test the function
sample_post = "I can't take this pain anymore. I just want to disappear."
print(predict_mental_risk(sample_post))