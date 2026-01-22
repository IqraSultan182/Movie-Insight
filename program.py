
import customtkinter as ctk
from predict_movie import check_movie
import joblib
from PIL import Image
import requests
from io import BytesIO
import os

ctk.set_appearance_mode("Dark")          
ctk.set_default_color_theme("dark-blue")  

class MoviePredictorGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🎬 MovieInsight")
        self.geometry("800x800")
        self.resizable(False, False)

        # Poster size (wide)
        self.poster_width = 650
        self.poster_height = 200  # laptop-style

        # Folder for movie posters
        self.poster_folder = "D:/Project_Ai/poster"

        # Default poster
        self.default_poster_path = os.path.join(self.poster_folder, "movies.jpg")
        if not os.path.exists(self.default_poster_path):
            # create a gray placeholder if default not found
            default_img = Image.new("RGB", (self.poster_width, self.poster_height), color="#555555")
        else:
            default_img = Image.open(self.default_poster_path)

        default_img = default_img.resize((self.poster_width, self.poster_height))
        self.default_poster_image = ctk.CTkImage(default_img, size=(self.poster_width, self.poster_height))
        self.current_poster_image = self.default_poster_image  # Start with default

        # Load model accuracy
        try:
            self.model_accuracy = joblib.load("model_accuracy.pkl")
            self.model_accuracy_percent = round(self.model_accuracy * 100, 2)
        except:
            self.model_accuracy_percent = "N/A"

        # --- Header ---
        self.header = ctk.CTkLabel(
            self,
            text="🎬 Movie Insight",
            font=ctk.CTkFont(family="Helvetica", size=24, weight="bold"),
            text_color="#FFD700"
        )
        self.header.pack(pady=15)

        # --- Input Frame ---
        self.input_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#2C2F33")
        self.input_frame.pack(padx=20, pady=10, fill="x")

        self.movie_entry = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="Enter Movie Name",
            width=500, height=35,
            font=ctk.CTkFont(size=14)
        )
        self.movie_entry.pack(pady=15, padx=20)

        self.btn_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.btn_frame.pack(pady=5)

        self.predict_btn = ctk.CTkButton(
            self.btn_frame,
            text="🔮 Predict Worth",
            command=self.predict,
            width=180,
            corner_radius=12,
            fg_color="#1abc9c",
            hover_color="#16a085"
        )
        self.predict_btn.pack(side="left", padx=10)

        self.clear_btn = ctk.CTkButton(
            self.btn_frame,
            text="🧹 Clear",
            command=self.clear,
            width=120,
            corner_radius=12,
            fg_color="#e74c3c",
            hover_color="#c0392b"
        )
        self.clear_btn.pack(side="left", padx=10)

        # --- Output Frame ---
        self.output_card = ctk.CTkFrame(self, corner_radius=20, fg_color="#1e1e2f")
        self.output_card.pack(padx=20, pady=20, fill="both", expand=True)

        # --- Poster Label ---
        self.poster_label = ctk.CTkLabel(self.output_card, image=self.current_poster_image, text="")
        self.poster_label.pack(pady=10)

        # --- Movie info labels ---
        self.title_label = ctk.CTkLabel(self.output_card, text="Title: ",
                                        font=ctk.CTkFont(size=16, weight="bold"),
                                        text_color="#FFD700")
        self.title_label.pack(anchor="w", pady=5, padx=20)

        self.rating_label = ctk.CTkLabel(self.output_card, text="Rating: ",
                                         font=ctk.CTkFont(size=14), text_color="#1abc9c")
        self.rating_label.pack(anchor="w", pady=5, padx=20)

        self.overview_label = ctk.CTkLabel(self.output_card, text="Overview: ",
                                           font=ctk.CTkFont(size=14), wraplength=750,
                                           justify="left", text_color="#ecf0f1")
        self.overview_label.pack(anchor="w", pady=5, padx=20)

        self.prediction_label = ctk.CTkLabel(self.output_card, text="Prediction: ",
                                             font=ctk.CTkFont(size=16, weight="bold"))
        self.prediction_label.pack(anchor="w", pady=5, padx=20)

        self.confidence_label = ctk.CTkLabel(self.output_card, text="Confidence: ",
                                             font=ctk.CTkFont(size=14), text_color="#f39c12")
        self.confidence_label.pack(anchor="w", pady=5, padx=20)

        self.accuracy_label = ctk.CTkLabel(self.output_card,
                                           text=f"Model Accuracy: {self.model_accuracy_percent}%",
                                           font=ctk.CTkFont(size=14), text_color="#3498db")
        self.accuracy_label.pack(anchor="w", pady=5, padx=20)

    def predict(self):
        movie_name = self.movie_entry.get().strip()
        if not movie_name:
            self.title_label.configure(text="Please enter a movie name!")
            return

        data = check_movie(movie_name)

        # --- Load Poster ---
        # Use data['title'] for matching poster file (assuming it's the exact title)
        poster_file = os.path.join(self.poster_folder, f"{data['title']}.jpg")
        print(f"Looking for poster: {poster_file}")  # Debug print
        if os.path.exists(poster_file):
            poster_img = Image.open(poster_file)
            print("Poster found and loaded.")  # Debug print
        else:
            poster_img = Image.open(self.default_poster_path)
            print("Poster not found, using default.")  # Debug print

        poster_img = poster_img.resize((self.poster_width, self.poster_height))
        self.current_poster_image = ctk.CTkImage(poster_img, size=(self.poster_width, self.poster_height))
        self.poster_label.configure(image=self.current_poster_image)
        self.poster_label.image = self.current_poster_image  # persistent reference

        # --- Update Labels ---
        self.title_label.configure(text=f"🎬 Title: {data['title']}")
        self.rating_label.configure(text=f"⭐ Rating: {data['rating']}")
        self.overview_label.configure(text=f"📝 Overview: {data['overview']}")
        self.prediction_label.configure(
            text=f"Prediction: {data['result']}",
            text_color="#27AE60" if data['result'] == "Worth Watching" else "#E74C3C"
        )
        self.confidence_label.configure(text=f"Confidence: {data['confidence']}%")
        self.accuracy_label.configure(text=f"Model Accuracy: {self.model_accuracy_percent}%")

    def clear(self):
        self.movie_entry.delete(0, "end")
        self.title_label.configure(text="Title: ")
        self.rating_label.configure(text="Rating: ")
        self.overview_label.configure(text="Overview: ")
        self.prediction_label.configure(text="Prediction: ", text_color="white")
        self.confidence_label.configure(text="Search Accuracy ")
        self.accuracy_label.configure(text=f"Model Accuracy: {self.model_accuracy_percent}%")
        # Reset poster to default
        self.current_poster_image = self.default_poster_image
        self.poster_label.configure(image=self.current_poster_image)
        self.poster_label.image = self.current_poster_image

if __name__ == "__main__":
    app = MoviePredictorGUI()
    app.mainloop()
