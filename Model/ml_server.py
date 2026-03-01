from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------- Load and prepare data ----------
df = pd.read_csv(
    r"C:\Program Files\Apache Software Foundation\Tomcat 11.0\webapps\SpotifyClone\python_iml\Dataset\songs.csv",
    encoding="ISO-8859-1"
)
df['Song_id'] = pd.to_numeric(df['Song_id'], errors='coerce').fillna(0).astype(int)

def duration_to_seconds(dur):
    try:
        parts = str(dur).split(':')
        if len(parts) == 2:
            return int(parts[0])*60 + int(parts[1])
        return int(dur)
    except:
        return 0

df['duration_sec'] = df['Duration'].apply(duration_to_seconds)

# One-hot encode categorical columns
categorical_cols = ['Genre', 'Mood', 'Language']
encoder = OneHotEncoder(handle_unknown="ignore")
encoded_cat = encoder.fit_transform(df[categorical_cols].astype(str)).toarray()

# Map popularity to numeric
pop_map = {'Very Low':10,'Low':30,'Medium':50,'High':70,'Very High':90}
df['popularity_num'] = df['Popularity'].map(pop_map).fillna(50)

# Combine numerical features
numerical = df[['duration_sec','Release_Year','popularity_num']].values
numerical = numerical * [0.5,0.3,0.7]
X = np.hstack([numerical, encoded_cat])
X = np.nan_to_num(X, nan=0.0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction
pca = PCA(n_components=min(50, X_scaled.shape[1]))
X_pca = pca.fit_transform(X_scaled)

# KNN for nearest neighbors
knn = NearestNeighbors(n_neighbors=6)
knn.fit(X_pca)

# ---------- Recommendation endpoint ----------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    try:
        song_id = int(data.get("song_id", 0))
    except:
        return jsonify({"reply": "Invalid song_id", "recommended": []})

    if song_id not in df['Song_id'].values:
        return jsonify({"reply": "Song not found", "recommended": []})

    song_index = df[df['Song_id'] == song_id].index[0]
    distances, indices = knn.kneighbors([X_pca[song_index]])
    recommended = []
    for idx in indices[0]:
        if idx != song_index:
            rec_song = {
                "Song_id": int(df.loc[idx, 'Song_id']),
                "Track_Name": str(df.loc[idx, 'Track_Name']),
                "Album": str(df.loc[idx, 'Album']),
                "Genre": str(df.loc[idx, 'Genre'])
            }
            recommended.append(rec_song)

    print(f"Received Song ID: {song_id}")
    print("Recommended songs:", [r["Song_id"] for r in recommended])

    return jsonify({
        "reply": f"Recommendations generated for song {song_id}",
        "recommended": recommended
    })

if __name__ == "__main__":
    print("✅ ML Server running at http://127.0.0.1:5000/")
    app.run(debug=True)