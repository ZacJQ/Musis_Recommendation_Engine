import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from joblib import dump, load  # Added import for joblib

base_dir = "/Users/zac/Codes/Music_Project/GIT_HUB/Musis_Recommendation_Engine"
file_path = "exploration/Data_collection/final_filtered_mood_list.csv"

dataset = pd.read_csv(os.path.join(base_dir,file_path))

try:
    dataset = dataset.rename(columns={"MOOD":"mood"})
    dataset.drop(columns=["Unnamed: 0"], inplace=True)
except Exception as e:
    print("Already column removed", e)

# Select relevant features
features = ['valence', 'popularity', 'year', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'duration_ms']

# Remove rows with missing values
dataset = dataset.dropna(subset=features)
dataset = dataset.drop(columns=['artist_name','track_name'])

# Select features again after one-hot encoding
features = dataset.columns.tolist()
X = dataset[features]
scaler = StandardScaler()
X_inference = scaler.fit_transform(X)

print(scaler.mean_ , scaler.scale_)

loaded_model = load('model.joblib')

predictions = loaded_model.predict(X_inference)
