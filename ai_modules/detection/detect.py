"""Debris Detection AI - classifies debris as decayed or in-orbit using orbital parameters."""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(MODULE_DIR, "space_decay.csv")
SEED = 42
TEST_FRACTION = 0.2
LABEL_COL = "decayed"

NUMERIC_FEATURES = [
    'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION',
    'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY',
    'PERIOD', 'APOAPSIS', 'PERIAPSIS'
]
CATEGORICAL_FEATURES = ['OBJECT_TYPE', 'RCS_SIZE']


def load_data():
    """Load the raw dataset from CSV."""
    if not os.path.exists(DATA_PATH):
        print(f"Warning: {DATA_PATH} not found")
        return None
    return pd.read_csv(DATA_PATH, low_memory=False)


def prepare_dataset():
    """Load data, encode features, and scale numerics."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df[LABEL_COL] = df['DECAY_DATE'].notna().astype(int)
    
    # Fill missing numeric values with median
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    
    # Encode categorical columns
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('UNKNOWN')
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    
    # Standardise numeric features
    scaler = StandardScaler()
    df[NUMERIC_FEATURES] = scaler.fit_transform(df[NUMERIC_FEATURES])
    
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[LABEL_COL]
    
    return X, y, encoders, scaler


def fit_model(X, y):
    """Train a Random Forest classifier on the dataset."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=200, random_state=SEED)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    
    return model


def train_detector():
    """Train and save the debris decay classifier."""
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found")
        print("Training requires the space_decay.csv dataset")
        return None
    
    X, y, encoders, scaler = prepare_dataset()
    decayed_count = y.sum()
    orbit_count = len(y) - decayed_count
    print(f"Total samples: {len(y)}, Decayed: {decayed_count}, Still in orbit: {orbit_count}")
    
    model = fit_model(X, y)
    
    joblib.dump(model, os.path.join(MODULE_DIR, "decay_classifier.pkl"))
    joblib.dump(encoders, os.path.join(MODULE_DIR, "label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(MODULE_DIR, "scaler.pkl"))
    print("Model and preprocessing saved.")
    
    return model


def load_model(path=None):
    """Load a trained model from disk."""
    if path is None:
        path = os.path.join(MODULE_DIR, "decay_classifier.pkl")
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_debris(model, features):
    """Predict whether debris has decayed."""
    if model is None:
        return {'error': 'No model provided'}
    
    try:
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][prediction]
        
        return {
            'prediction': int(prediction),
            'status': 'Decayed' if prediction == 1 else 'Still in orbit',
            'probability': float(confidence)
        }
    except Exception as e:
        return {'error': str(e)}


def run_detection_demo():
    """Demonstrate the debris classifier on sample data."""
    print("Debris Detection AI Demo")
    print("=" * 50)
    
    model_path = os.path.join(MODULE_DIR, "decay_classifier.pkl")
    encoders_path = os.path.join(MODULE_DIR, "label_encoders.pkl")
    scaler_path = os.path.join(MODULE_DIR, "scaler.pkl")
    
    required_files = [model_path, encoders_path, scaler_path]
    if not all(os.path.exists(p) for p in required_files):
        print("Model not found. Run training first.")
        return None
    
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)
    
    sample = pd.DataFrame([{
        'MEAN_MOTION': 2.5,
        'ECCENTRICITY': 0.01,
        'INCLINATION': 98.5,
        'RA_OF_ASC_NODE': 50.0,
        'ARG_OF_PERICENTER': 120.0,
        'MEAN_ANOMALY': 200.0,
        'PERIOD': 95.0,
        'APOAPSIS': 400.0,
        'PERIAPSIS': 390.0,
        'OBJECT_TYPE': 'DEBRIS',
        'RCS_SIZE': 'SMALL'
    }])
    
    for col in CATEGORICAL_FEATURES:
        sample[col] = encoders[col].transform(sample[col])
    
    sample[NUMERIC_FEATURES] = scaler.transform(sample[NUMERIC_FEATURES])
    
    result = model.predict(sample)[0]
    confidence = model.predict_proba(sample)[0][result]
    
    print("Predicted decay status:", "Decayed" if result == 1 else "Still in orbit")
    print("Probability:", confidence)
    
    return {'prediction': int(result), 'probability': float(confidence)}


if __name__ == "__main__":
    run_detection_demo()
