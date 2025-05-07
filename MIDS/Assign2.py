import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords  # Import stopwords
nltk.download('stopwords')  # Download stopwords once
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Sentiment140 dataset
df = pd.read_csv(
    r"C:\Users\abhis\OneDrive\Desktop\My Playground\Python\training.1600000.processed.noemoticon.csv\training.1600000.processed.noemoticon.csv",
    encoding="ISO-8859-1",
    names=["target", "id", "date", "flag", "user", "text"]
)

# Keep only text and target
df = df[['text', 'target']]
df['sentiment'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
df = df.sample(n=50000, random_state=42)

# Define stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

df['cleaned_text'] = df['text'].apply(preprocess_text)

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski')  # Use a valid metric
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))