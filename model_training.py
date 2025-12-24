import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 1. Load Dataset
try:
    df = pd.read_csv("dataset.csv")
    df.dropna(inplace=True) 
    print("✅ Dataset loaded successfully.")
    print(f"📊 Total samples: {len(df)}")
except FileNotFoundError:
    print("❌ Error: 'dataset.csv' not found.")
    exit()

# 2. Prepare Data
X = df['text']
y = df['mood']

# 3. Split Data
# Using a smaller test size (0.15) so more data is used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# 4. Create Pipeline with Adjusted Vectorizer
# ✨ FIX: Changed min_df to 1 (uses every word)
# ✨ FIX: Removed stop_words='english' to capture short phrases like "I am sad" better
model = make_pipeline(
    CountVectorizer(ngram_range=(1, 2), min_df=1), 
    MultinomialNB()
)

# 5. Train
print("⚙️ Training model...")
model.fit(X_train, y_train)

# 6. Evaluate
predicted = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predicted)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# 7. Save
with open("mood_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("💾 Model saved as 'mood_model.pkl'")