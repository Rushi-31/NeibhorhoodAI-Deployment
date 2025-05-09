from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
import os

app = Flask(__name__)

# -----------------------------
# Load Neighborhood Dataset
# -----------------------------
try:
    dataset_path =  r'datasets/pd_copy.csv'
    neighborhood_df = pd.read_csv(dataset_path)
except FileNotFoundError:
    neighborhood_df = pd.DataFrame()

# -----------------------------
# Helper Function
# -----------------------------
def parse_income(value):
    if pd.isna(value):
        return np.nan
    value = value.replace(",", "").replace("+", "")
    if "-" in value:
        low, high = map(int, value.split("-"))
        return (low + high) / 2
    return int(value)

# -----------------------------
# Preprocess Neighborhood Dataset
# -----------------------------
if not neighborhood_df.empty:
    neighborhood_df['Income_Estimate'] = neighborhood_df['General Income (INR)'].apply(parse_income)
    neighborhood_df = neighborhood_df.dropna(subset=['Income_Estimate'])

    if 'Sublocality' in neighborhood_df.columns:
        neighborhood_df['Combined_Neighborhood'] = neighborhood_df['Neighborhood'].astype(str).str.strip() + ", " + neighborhood_df['Sublocality'].astype(str).str.strip()
    else:
        neighborhood_df['Combined_Neighborhood'] = neighborhood_df['Neighborhood'].astype(str).str.strip()

    input_features = ['Professions', 'Income_Estimate', 'Community Sentiment', 'Public Transport']
    output_features = ['Combined_Neighborhood', 'Crime Rate', 'Schools Nearby', 'Hospitals Nearby', 'Professions']

    categorical_features = ['Professions', 'Community Sentiment', 'Public Transport']
    numerical_features = ['Income_Estimate']

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

    X_processed = preprocessor.fit_transform(neighborhood_df[input_features])
    knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
    knn_model.fit(X_processed)

# -----------------------------
# ROUTES
# -----------------------------

@app.route('/')
def home():
    return "Hello, I am ready!"

@app.route('/recommend', methods=['POST'])
def recommend():
    if neighborhood_df.empty:
        return jsonify({"error": "Neighborhood data not found."}), 500

    try:
        user_input = request.get_json()
        user_df = pd.DataFrame([user_input])
        user_processed = preprocessor.transform(user_df[input_features])
        distances, indices = knn_model.kneighbors(user_processed, n_neighbors=5)

        results = neighborhood_df.iloc[indices[0]][output_features].reset_index(drop=True)
        results = results.rename(columns={"Combined_Neighborhood": "Neighborhood"})
        results['Similarity Score'] = (1 / (1 + distances[0])).round(3)
        return jsonify(results.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()