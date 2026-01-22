import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# === CONFIG ===
INPUT_CSV = "train_dataset.csv"        # Your cleaned dataset
SAMPLE_N = 200000                       # Adjust if low RAM (try 100000)
TFIDF_MAX_FEATURES = 5000
MODEL_OUT = "movie_model_tfidf_lr.pkl"
VECT_OUT = "tfidf_vectorizer.pkl"
ACCURACY_OUT = "model_accuracy.pkl"    # File to save model accuracy

# === 1. Load Sample ===
print("📂 Loading sample data...")
df = pd.read_csv(INPUT_CSV, encoding='latin1')
print("✅ Data loaded successfully!")
print(df.head())

sample_df = df.sample(n=min(SAMPLE_N, len(df)), random_state=42)

# === 2. Clean Missing Values ===
sample_df = sample_df.dropna(subset=['clean_overview', 'worth_watching'])
X = sample_df['clean_overview'].astype(str)
y = sample_df['worth_watching'].astype(int)

# === 3. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. TF-IDF Vectorization ===
print("🔠 Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === 5. Model Training ===
print("🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=300, n_jobs=-1, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# === 6. Evaluate ===
print("📊 Evaluating model...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 7. Save Model + Vectorizer + Accuracy ===
joblib.dump(model, MODEL_OUT)
joblib.dump(vectorizer, VECT_OUT)
joblib.dump(accuracy, ACCURACY_OUT)  # Save accuracy for GUI

print("\n💾 Model saved as:", MODEL_OUT)
print("💾 Vectorizer saved as:", VECT_OUT)
print("💾 Model accuracy saved as:", ACCURACY_OUT)
print("\n🎉 Training complete! You're ready for prediction.")
