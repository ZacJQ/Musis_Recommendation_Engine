import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
dataset = pd.read_csv("/Users/zac/Codes/Music_Project/GIT_HUB/Musis_Recommendation_Engine/data/testing/testing.csv")

# Select relevant features
features = ['valence', 'popularity', 'year', 'genre', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'duration_ms']

# Remove rows with missing values
dataset = dataset.dropna(subset=features+['mood'])

dataset = dataset.rename(columns={"MOOD":"mood"})
dataset.drop(columns=["Unnamed: 0"], inplace=True)

# # Encode categorical variables if any
# dataset = pd.get_dummies(dataset, columns=['genre'])

# Split dataset into features and target
X = dataset[features]
y = dataset['mood']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train classifier
classifier.fit(X_train, y_train)

# Predict on test set
y_pred = classifier.predict(X_test)

# Evaluate classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
