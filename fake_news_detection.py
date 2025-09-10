# fake_news_detection.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
# Load datasets
fake = pd.read_csv("C:/Users/As/Downloads/Fake.csv")
real = pd.read_csv("C:/Users/As/Downloads/True.csv")
 # Add labels
fake["label"] = 0  # FAKE
real["label"] = 1  # REAL

 # Combine the datasets
df = pd.concat([fake, real], axis=0)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
# Use only 'text' and 'label'
X = df['text']
y = df['label']
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Passive Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)
 # Predict
y_pred = model.predict(X_test_tfidf) 
 # Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# Function to predict whether a news is fake or real
def predict_news(news):
    transformed = vectorizer.transform([news])
    prediction = model.predict(transformed)
    return "REAL" if prediction[0] == 1 else "FAKE"
# Use it
sample_news = input("\nEnter the news content to check if it's FAKE or REAL:\n")
result = predict_news(sample_news)
print(f"\nPrediction: {result}")

 


import pickle

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
