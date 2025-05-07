import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Simple stopword list (offline version)
stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours", "he", "him", 
    "his", "she", "her", "hers", "it", "its", "they", "them", "their", "what", "which", 
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", 
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", 
    "shall", "can", "could", "should", "may", "might", "must", "very", "a", "an", "the", "and", "or"
])

# Simple tokenizer using regex
def simple_tokenize(text):
    return re.findall(r'\b[a-z]{2,}\b', text.lower())

# Preprocessing function
def preprocess(text):
    tokens = simple_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words])

# Sample dataset
documents = [
    ("I love this product, it is fantastic and great!", "positive"),
    ("This is a terrible experience, I hate it!", "negative"),
    ("I am so happy with the service, excellent support!", "positive"),
    ("The movie was horrible, I dislike it!", "negative"),
    ("It was a good day, everything went well!", "positive"),
    ("Awful customer service, very bad experience!", "negative")
]

# Prepare dataset
texts, labels = zip(*documents)
texts = [preprocess(text) for text in texts]

# Convert text to feature vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = [1 if label == "positive" else 0 for label in labels]

# Train Naïve Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, y)

# Function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess(text)
    text_vector = vectorizer.transform([processed_text])
    prediction = classifier.predict(text_vector)
    return "positive" if prediction[0] == 1 else "negative"

# Debugging
print("Processed texts:", texts)
print("Vocabulary:", vectorizer.get_feature_names_out())

# Test
sample_text = "The product is great and I enjoy using it!"
print(f"Sentiment: {predict_sentiment(sample_text)}")
