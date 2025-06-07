import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import joblib
import os

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove @mentions
    text = re.sub(r'#', '', text)  # Remove # symbol
    text = re.sub(r'RT[\s]+', '', text)  # Remove RT
    text = re.sub(r'https?:\/\/\S+', '', text)  # Remove hyperlinks
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    tokens = nltk.word_tokenize(text)  # Tokenize
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords.words('english')]  # Stemming and stopword removal
    return ' '.join(tokens)

def train_model():
    print("Loading dataset...")
    dataset = load_dataset("tweet_eval", "sentiment")
    
    # Convert to pandas DataFrame for easier processing
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])

    # Combine train, validation, and test sets for training
    df = pd.concat([train_df, val_df, test_df])

    # Map labels to meaningful names
    # 0: negative, 1: neutral, 2: positive
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['label'] = df['label'].map(label_map)

    print("Preprocessing text data...")
    df['text'] = df['text'].apply(preprocess_text)

    X = df['text']
    y = df['label']

    # Split data for training the final model (optional, could use all data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Logistic Regression model...")
    # Create a pipeline with TF-IDF Vectorizer and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)), # Limiting features for simplicity
        ('clf', LogisticRegression(solver='liblinear', multi_class='auto', random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    print("Evaluating model on test set...")
    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained pipeline
    model_path = 'social-media-nlp/model/sentiment_pipeline.joblib'
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model() 