"""
MoodMate Backend - Single File Implementation
Emotion-based music recommendation system
"""

import os
import time
import logging
import random
import joblib
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
from transformers import pipeline

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Get environment variables
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "39a01536237b4e938f5ab254d652e726")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "81e04b3e8d4b4045abddfae5ca93e086")
MODEL_PATH = os.getenv("MODEL_PATH", "./models")

# Emotion labels from GoEmotions dataset
EMOTION_LABELS = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
    19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
    23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}

# Emotion to music mood mapping
EMOTION_MUSIC_MAP = {
    "admiration": {"query": "inspiring upbeat", "suggestions": ["Set new goals", "Read inspiring content"]},
    "amusement": {"query": "funny comedy", "suggestions": ["Watch comedy", "Share jokes with friends"]},
    "joy": {"query": "happy upbeat dance", "suggestions": ["Dance to music", "Spend time with loved ones"]},
    "love": {"query": "romantic love songs", "suggestions": ["Write a love letter", "Plan a romantic evening"]},
    "excitement": {"query": "energetic dance electronic", "suggestions": ["Try new adventure", "Share your excitement"]},
    "gratitude": {"query": "thankful peaceful", "suggestions": ["Write gratitude journal", "Thank someone special"]},
    "optimism": {"query": "hopeful motivational", "suggestions": ["Make positive plans", "Visualize your goals"]},
    "anger": {"query": "aggressive rock metal", "suggestions": ["Do intense exercise", "Practice deep breathing"]},
    "sadness": {"query": "melancholy sad ballads", "suggestions": ["Allow yourself to feel", "Reach out to friends"]},
    "fear": {"query": "calming ambient", "suggestions": ["Practice breathing exercises", "Seek support"]},
    "surprise": {"query": "unexpected eclectic", "suggestions": ["Embrace the unexpected", "Stay curious"]},
    "neutral": {"query": "chill lofi ambient", "suggestions": ["Enjoy the moment", "Practice mindfulness"]},
}

# Fill in missing emotions with neutral mapping
for emotion in EMOTION_LABELS.values():
    if emotion not in EMOTION_MUSIC_MAP:
        EMOTION_MUSIC_MAP[emotion] = EMOTION_MUSIC_MAP["neutral"]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    music_limit: int = Field(default=10, ge=1, le=20)

class EmotionResult(BaseModel):
    predicted_emotion: str
    confidence: float
    top_emotions: Dict[str, float]

class MusicTrack(BaseModel):
    name: str
    artist: str
    url: str

class AnalysisResponse(BaseModel):
    emotion: EmotionResult
    music_tracks: List[MusicTrack]
    suggestions: List[str]
    processing_time: float

# ============================================================================
# GLOBAL VARIABLES & MODEL STORAGE
# ============================================================================

# Global variables to store loaded models and services
emotion_model = None
vectorizer = None
spotify_client = None
llm_pipeline = None

# ============================================================================
# EMOTION DETECTION
# ============================================================================

class EmotionDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train_and_save(self):
        """Train model on GoEmotions dataset and save it"""
        logger.info("Loading GoEmotions dataset...")
        dataset = load_dataset("go_emotions", "simplified")

        # Prepare data
        texts = list(dataset["train"]["text"])
        labels = [label_list[0] if label_list else 27 for label_list in dataset["train"]["labels"]]

        logger.info(f"Training on {len(texts)} samples...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Train vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english', max_features=5000, ngram_range=(1, 2)
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train model
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.model.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.3f}")

        # Save models
        os.makedirs(MODEL_PATH, exist_ok=True)
        joblib.dump(self.model, f"{MODEL_PATH}/emotion_model.joblib")
        joblib.dump(self.vectorizer, f"{MODEL_PATH}/vectorizer.joblib")
        logger.info("Models saved!")

        return accuracy

    def load_models(self):
        """Load trained models"""
        try:
            self.model = joblib.load(f"{MODEL_PATH}/emotion_model.joblib")
            self.vectorizer = joblib.load(f"{MODEL_PATH}/vectorizer.joblib")
            logger.info("Models loaded successfully")
            return True
        except:
            logger.warning("Models not found, will train new ones")
            return False

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict emotion from text"""
        if not self.model or not self.vectorizer:
            raise ValueError("Models not loaded")

        # Vectorize text
        text_vec = self.vectorizer.transform([text])

        # Get predictions
        probabilities = self.model.predict_proba(text_vec)[0]
        predicted_idx = np.argmax(probabilities)

        # Create results
        predicted_emotion = EMOTION_LABELS[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # Top 3 emotions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_emotions = {
            EMOTION_LABELS[idx]: float(probabilities[idx]) 
            for idx in top_indices
        }

        return predicted_emotion, confidence, top_emotions

# ============================================================================
# SPOTIFY SERVICE
# ============================================================================

class SpotifyService:
    def __init__(self):
        self.client = None

    def initialize(self):
        """Initialize Spotify client"""
        try:
            auth_manager = SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET
            )
            self.client = spotipy.Spotify(auth_manager=auth_manager)
            # Test connection
            self.client.search(q="test", type="track", limit=1)
            logger.info("Spotify client initialized")
            return True
        except Exception as e:
            logger.error(f"Spotify initialization failed: {e}")
            return False

    def get_music_recommendations(self, emotion: str, limit: int = 10) -> List[Dict]:
        """Get music recommendations based on emotion"""
        if not self.client:
            return []

        try:
            # Get search query for emotion
            mood_data = EMOTION_MUSIC_MAP.get(emotion, EMOTION_MUSIC_MAP["neutral"])
            query = mood_data["query"]

            # Search Spotify
            results = self.client.search(q=query, type="track", limit=limit*2, market="US")

            tracks = []
            for item in results["tracks"]["items"]:
                if not item:
                    continue

                track = {
                    "name": item["name"],
                    "artist": ", ".join([artist["name"] for artist in item["artists"]]),
                    "url": item["external_urls"]["spotify"]
                }
                tracks.append(track)

                if len(tracks) >= limit:
                    break

            # Shuffle for variety
            random.shuffle(tracks)
            return tracks[:limit]

        except Exception as e:
            logger.error(f"Spotify search failed: {e}")
            return []

# ============================================================================
# LLM SERVICE (SIMPLIFIED)
# ============================================================================

def get_activity_suggestions(emotion: str, user_text: str) -> List[str]:
    """Get activity suggestions (simplified version)"""
    # Use predefined suggestions based on emotion
    base_suggestions = EMOTION_MUSIC_MAP.get(emotion, EMOTION_MUSIC_MAP["neutral"])["suggestions"]

    # Add some general suggestions
    general_suggestions = [
        "Take a walk outside",
        "Call a friend or family member", 
        "Practice deep breathing",
        "Listen to your favorite music",
        "Do something creative"
    ]

    # Combine and randomize
    all_suggestions = base_suggestions + general_suggestions
    random.shuffle(all_suggestions)

    return all_suggestions[:3]

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="MoodMate Backend",
    description="Emotion-based music recommendation system",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    global emotion_model, vectorizer, spotify_client, llm_pipeline

    logger.info("Starting MoodMate backend...")

    # Initialize emotion detector
    emotion_model = EmotionDetector()
    if not emotion_model.load_models():
        logger.info("Training new emotion detection model...")
        accuracy = emotion_model.train_and_save()
        emotion_model.load_models()

    # Initialize Spotify
    spotify_client = SpotifyService()
    spotify_client.initialize()

    logger.info("MoodMate backend ready!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to MoodMate Backend!",
        "docs": "/docs",
        "health": "/health"
    }
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "emotion_model": emotion_model is not None and emotion_model.model is not None,
            "spotify": spotify_client is not None and spotify_client.client is not None
        }
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_mood(request: AnalysisRequest):
    """Main endpoint: analyze emotion and get music recommendations"""
    start_time = time.time()

    try:
        # Check if services are ready
        if not emotion_model or not emotion_model.model:
            raise HTTPException(status_code=503, detail="Emotion detection service not ready")

        # Analyze emotion
        predicted_emotion, confidence, top_emotions = emotion_model.predict(request.text)

        # Get music recommendations
        music_tracks = []
        if spotify_client and spotify_client.client:
            track_data = spotify_client.get_music_recommendations(
                predicted_emotion, request.music_limit
            )
            music_tracks = [MusicTrack(**track) for track in track_data]

        # Get activity suggestions
        suggestions = get_activity_suggestions(predicted_emotion, request.text)

        processing_time = time.time() - start_time

        return AnalysisResponse(
            emotion=EmotionResult(
                predicted_emotion=predicted_emotion,
                confidence=confidence,
                top_emotions=top_emotions
            ),
            music_tracks=music_tracks,
            suggestions=suggestions,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.post("/emotion-only")
async def analyze_emotion_only(request: AnalysisRequest):
    """Analyze emotion only (no music recommendations)"""
    try:
        if not emotion_model or not emotion_model.model:
            raise HTTPException(status_code=503, detail="Emotion detection service not ready")

        predicted_emotion, confidence, top_emotions = emotion_model.predict(request.text)

        return {
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "top_emotions": top_emotions
        }

    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Emotion analysis failed")

@app.get("/emotions")
async def list_emotions():
    """List all supported emotions"""
    return {
        "emotions": list(EMOTION_LABELS.values()),
        "total": len(EMOTION_LABELS)
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
