import pandas as pd
import joblib

# === Load Model, Vectorizer, Dataset and Accuracy ===
print("🔄 Loading model, vectorizer, dataset and accuracy...")

try:
    model = joblib.load("movie_model_tfidf_lr.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    accuracy = joblib.load("model_accuracy.pkl")  # overall model accuracy
    df = pd.read_csv("train_dataset.csv", encoding='latin1')
except FileNotFoundError as e:
    print("❌ File not found:", e)
    exit()

print("✅ All files loaded successfully!\n")

def check_movie(movie_name):
    """
    Predict if a movie is Worth Watching or Not, return its title, rating,
    overview, prediction, per-movie confidence, and overall model accuracy
    """
    movie = df[df['title'].str.lower() == movie_name.lower()]
    
    if movie.empty:
        # Movie not found
        return {
            "title": movie_name,
            "rating": "N/A",
            "overview": "Movie not found in database.",
            "result": "Unknown",
            "confidence": "N/A",
            "model_accuracy": round(accuracy*100, 2)
        }

    title = movie['title'].values[0]
    rating = movie['vote_average'].values[0]
    overview = movie['overview'].values[0]

    # Convert overview to TF-IDF
    overview_tfidf = vectorizer.transform([overview])

    # Predict per-movie probability
    prob = model.predict_proba(overview_tfidf)[0][1] * 100
    prediction = model.predict(overview_tfidf)[0]
    result = "Worth Watching" if prediction == 1 else "Not Worth Watching"

    return {
        "title": title,
        "rating": rating,
        "overview": overview[:300] + ("..." if len(overview) > 300 else ""),
        "result": result,
        "confidence": round(prob, 2),
        "model_accuracy": round(accuracy*100, 2)
    }

# === Main: Ask user for movie name ===
if __name__ == "__main__":
    print("🎬 Movie Worth-Watching Predictor\n")
    movie_name = input("Enter the movie name: ").strip()

    data = check_movie(movie_name)

    print("\n--- Movie Prediction Result ---")
    print(f"Title           : {data['title']}")
    print(f"Rating          : {data['rating']}")
    print(f"Overview        : {data['overview']}")
    print(f"Prediction      : {data['result']}")
    print(f"Confidence      : {data['confidence']}%")
    print(f"Model Accuracy  : {data['model_accuracy']}%")
