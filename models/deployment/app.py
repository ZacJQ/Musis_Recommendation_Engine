import gradio as gr
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd
from nltk.util import ngrams
from typing import List
from typing import List
from sklearn.cluster import KMeans
import random
import numpy as np

# Load the CSV file
base_dir = "/Users/zac/Codes/Music_Project/GIT_HUB/Musis_Recommendation_Engine"
file_path = 'exploration/Data_collection/final_filtered_mood_list.csv'
df = pd.read_csv(os.path.join(base_dir,file_path))

# Remove duplicates
df = df.drop_duplicates()
df_temp = df
df_new = df 

# Fit the scaler specifically for the tempo column
tempo_scaler = StandardScaler()
df_new['tempo'] = tempo_scaler.fit_transform(df[['tempo']])

popularity_scaler = StandardScaler()
df_new['popularity'] = popularity_scaler.fit_transform(df[['popularity']])

# Normalize numerical features
numerical_features = ['duration_ms', 'popularity', 'danceability', 'energy', 'loudness', 
                      'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                      'valence', 'tempo']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Encode the 'mood' as a categorical feature
label_encoder = LabelEncoder()
df['mood_encoded'] = label_encoder.fit_transform(df['mood'])

# Define the features for the k-NN model
features = ['duration_ms', 'popularity', 'danceability', 'energy', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'mood_encoded']

# Train the k-NN model
knn_model = NearestNeighbors(n_neighbors=5, metric='manhattan', leaf_size=30, )  # euclidean
knn_model.fit(df[features])


def recommend_songs(mood: str | None="HAPPY", 
                    year_range_min: int| None=None,year_range_max: int| None=None, 
                    tempo_range_min:int | None=None, tempo_range_max:int | None=None, 
                    exclude_artists: List[str] |None = None, 
                    randomize: bool | None=False, 
                    n_recommendations:int | None=10, 
                    deviation:float | None=0.1,
                    popularity_min:int | None=0,
                    popularity_max:int | None=100,
                    )-> List[dict]:
    """
    Song recommendation engine
    """
    year_range = [year_range_min, year_range_max]
    tempo_range = [tempo_range_min, tempo_range_max]
    popularity = [popularity_min, popularity_max]
    mood_encoded = label_encoder.transform([mood])[0]
    filtered_df = df[df['mood_encoded'] == mood_encoded]

    if year_range:
        filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
    if tempo_range:
        normalized_tempo_range = tempo_scaler.transform([[tempo_range[0]], [tempo_range[1]]]).flatten()
        filtered_df = filtered_df[(filtered_df['tempo'] >= normalized_tempo_range[0]) & (filtered_df['tempo'] <= normalized_tempo_range[1])]
        # filtered_df = filtered_df[(filtered_df['tempo'] >= tempo_range[0]) & (filtered_df['tempo'] <= tempo_range[1])]
    if popularity:
        normalized_popularity_range = popularity_scaler.transform([[popularity[0]], [popularity[1]]]).flatten()
        filtered_df = filtered_df[(filtered_df['popularity'] >= normalized_popularity_range[0]) & (filtered_df['popularity'] <= normalized_popularity_range[1])]
    if exclude_artists:
        filtered_df = filtered_df[~filtered_df['artist_name'].isin(exclude_artists)]
    if filtered_df.empty:
        filtered_df = df[df['mood_encoded'] == mood_encoded]

    # If the filtered_df is still empty, then return an empty recommendations list
    if filtered_df.empty:
        return []

    # Assuming filtered_df[features] * weights has been calculated
    adjusted_features = filtered_df[features] * weights

    # Define mean and standard deviation for Gaussian noise
    # mean = 0
    # std_deviation = 0.1  # You can adjust this value as needed

    # # Generate random Gaussian noise with the same shape as adjusted_features
    # noise = np.random.normal(mean, std_deviation, adjusted_features.shape)
    # adjusted_features_with_noise = adjusted_features + noise

    weights = [1] * len(features) 
    weights = [weight * (1 + deviation) for weight in weights]
    adjusted_features = filtered_df[features] * weights
    
    if randomize:
        recommendations = filtered_df.sample(n=n_recommendations)
    else:
        knn_distances, knn_indices = knn_model.kneighbors(adjusted_features, n_neighbors=n_recommendations)
        recommendations = df.iloc[knn_indices[0]]
    
    return recommendations[['artist_name', 'track_name', 'year', 'tempo', 'mood']].to_dict(orient='records')


# Find similar songs function
def find_similar_songs(input_song: str | None=None, 
                       artist_name: str | None=None,
                       n_similar: int | None=3,
                       mood: str | None=False, 
                       year_range_min: int| None=False,year_range_max: int| None=False, 
                       tempo_range_min:int | None=False, tempo_range_max:int | None=False, 
                       include_artists: List[str] |None = None, 
                       randomize: bool | None=False, 
                       deviation:float | None=0.1,
                       popularity_min:int | None=0,
                       popularity_max:int | None=100,
                       ):
    """ 
    Finds Simlar songs
    """
    year_range = [year_range_min, year_range_max]
    tempo_range = [tempo_range_min, tempo_range_max]
    popularity = [popularity_min, popularity_max]
    if mood:
        mood_encoded = label_encoder.transform([mood])[0]
        filtered_df = df[df['mood_encoded'] == mood_encoded]
    if input_song == None and artist_name== None:
        return "Song name cannot be empty"
    else:
        # input_features = df[df['track_name'] == input_song & df['artist_name'] == artist_name][features]
        input_features = _search_songs(song_name=input_song , artist_name=artist_name)[features]

        if input_features.empty:
            return []
        knn_distances, knn_indices = knn_model.kneighbors(input_features, n_neighbors=n_similar)
        similar_songs = df.iloc[knn_indices[0]]
        filtered_df = similar_songs

        if mood:
            mood_encoded = label_encoder.transform([mood])[0]
            filtered_df = df[df['mood_encoded'] == mood_encoded]
        if year_range:
            filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
        if tempo_range:
            normalized_tempo_range = tempo_scaler.transform([[tempo_range[0]], [tempo_range[1]]]).flatten()
            filtered_df = filtered_df[(filtered_df['tempo'] >= normalized_tempo_range[0]) & (filtered_df['tempo'] <= normalized_tempo_range[1])]
            # filtered_df = filtered_df[(filtered_df['tempo'] >= tempo_range[0]) & (filtered_df['tempo'] <= tempo_range[1])]
        if popularity:
            normalized_popularity_range = popularity_scaler.transform([[popularity[0]], [popularity[1]]]).flatten()
            filtered_df = filtered_df[(filtered_df['popularity'] >= normalized_popularity_range[0]) & (filtered_df['popularity'] <= normalized_popularity_range[1])]
        if include_artists:
            filtered_df = filtered_df[~filtered_df['artist_name'].isin(include_artists)]
        if filtered_df.empty:
            filtered_df = df[df['mood_encoded'] == mood_encoded]

    return filtered_df[['artist_name', 'track_name', 'year', 'tempo', 'mood']].to_dict(orient='records')

# Cluster similar songs function
def cluster_songs(mood: str | None="RANDOM", 
                  n_clusters: int | None=2, 
                  is_random: bool | None=False,
                  ):
    if is_random:
        random_state = random.randint(0,100)
    else:
        random_state = 42
    
    if mood != "RANDOM":
        mood_encoded = label_encoder.transform([mood])[0]
        filtered_df = df[df['mood_encoded'] == mood_encoded]
    else:
        filtered_df = df
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    filtered_df['cluster'] = kmeans.fit_predict(filtered_df[features])
    
    clusters = []
    for cluster in range(n_clusters):
        cluster_songs = filtered_df[filtered_df['cluster'] == cluster]
        clusters.append(cluster_songs[['artist_name', 'track_name', 'year', 'tempo', 'mood']].to_dict(orient='records'))
    return clusters


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def search_songs_old(song_name: str, artist_name: str)-> List[dict]:
    filtered_df = df
    filtered_df = filtered_df[(filtered_df['track_name'] == song_name) & (filtered_df['artist_name'] == artist_name)]
    return filtered_df[['artist_name', 'track_name', 'year', 'tempo', 'mood']].to_dict(orient='records')

def _search_songs(song_name: str, artist_name: str) -> pd.DataFrame:
    """
    To send matching song
    """
    # Preprocess the input strings if they are not empty
    song_name = song_name.lower() if song_name else ""
    artist_name = artist_name.lower() if artist_name else ""

    # Calculate N-grams for song name and artist name
    song_ngrams = set(ngrams(song_name, 3)) if song_name else set()
    artist_ngrams = set(ngrams(artist_name, 3)) if artist_name else set()

    # Handle possible NaN values in the DataFrame and calculate Jaccard similarity scores
    df['song_similarity'] = df['track_name'].apply(
        lambda x: jaccard_similarity(song_ngrams, set(ngrams(str(x).lower(), 3))) if pd.notna(x) else 0
    )
    df['artist_similarity'] = df['artist_name'].apply(
        lambda x: jaccard_similarity(artist_ngrams, set(ngrams(str(x).lower(), 3))) if pd.notna(x) else 0
    )

    # Filter the DataFrame based on similarity scores
    if song_name and artist_name:
        filtered_df = df[(df['song_similarity'] > 0.8) & (df['artist_similarity'] > 0.8)]
    elif song_name:
        filtered_df = df[df['song_similarity'] > 0.8]
    elif artist_name:
        filtered_df = df[df['artist_similarity'] > 0.8]
    else:
        return []

    # Return the filtered results as a list of dictionaries
    return filtered_df

def search_songs(song_name: str, artist_name: str) -> List[dict]:
    """
    To find matching song
    """
    # Preprocess the input strings if they are not empty
    song_name = song_name.lower() if song_name else ""
    artist_name = artist_name.lower() if artist_name else ""

    # Calculate N-grams for song name and artist name
    song_ngrams = set(ngrams(song_name, 3)) if song_name else set()
    artist_ngrams = set(ngrams(artist_name, 3)) if artist_name else set()

    # Handle possible NaN values in the DataFrame and calculate Jaccard similarity scores
    df['song_similarity'] = df['track_name'].apply(
        lambda x: jaccard_similarity(song_ngrams, set(ngrams(str(x).lower(), 3))) if pd.notna(x) else 0
    )
    df['artist_similarity'] = df['artist_name'].apply(
        lambda x: jaccard_similarity(artist_ngrams, set(ngrams(str(x).lower(), 3))) if pd.notna(x) else 0
    )

    # Filter the DataFrame based on similarity scores
    if song_name and artist_name:
        filtered_df = df[(df['song_similarity'] > 0.8) & (df['artist_similarity'] > 0.8)]
    elif song_name:
        filtered_df = df[df['song_similarity'] > 0.8]
    elif artist_name:
        filtered_df = df[df['artist_similarity'] > 0.8]
    else:
        return []

    # Return the filtered results as a list of dictionaries
    return filtered_df[['artist_name', 'track_name', 'year', 'tempo', 'mood']].to_dict(orient='records')

def predict(song_name):
    return f"{song_name}"

# JavaScript code for autocomplete functionality
# JavaScript code for autocomplete functionality
javascript_code = """
<script>
var song_names = {};  // Replace {} with your list of song names

function autocomplete(inp, arr) {
    var currentFocus;
    inp.addEventListener("input", function(e) {
        var a, b, i, val = this.value;
        closeAllLists();
        if (!val) { return false;}
        currentFocus = -1;
        a = document.createElement("DIV");
        a.setAttribute("id", this.id + "autocomplete-list");
        a.setAttribute("class", "autocomplete-items");
        this.parentNode.appendChild(a);
        for (i = 0; i < arr.length; i++) {
          if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
            b = document.createElement("DIV");
            b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
            b.innerHTML += arr[i].substr(val.length);
            b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
            b.addEventListener("click", function(e) {
                inp.value = this.getElementsByTagName("input")[0].value;
                closeAllLists();
            });
            a.appendChild(b);
          }
        }
    });
    inp.addEventListener("keydown", function(e) {
        var x = document.getElementById(this.id + "autocomplete-list");
        if (x) x = x.getElementsByTagName("div");
        if (e.keyCode == 40) {
          currentFocus++;
          addActive(x);
        } else if (e.keyCode == 38) {
          currentFocus--;
          addActive(x);
        } else if (e.keyCode == 13) {
          e.preventDefault();
          if (currentFocus > -1) {
            if (x) x[currentFocus].click();
          }
        }
    });
    function addActive(x) {
      if (!x) return false;
      removeActive(x);
      if (currentFocus >= x.length) currentFocus = 0;
      if (currentFocus < 0) currentFocus = (x.length - 1);
      x[currentFocus].classList.add("autocomplete-active");
    }
    function removeActive(x) {
      for (var i = 0; i < x.length; i++) {
        x[i].classList.remove("autocomplete-active");
      }
    }
    function closeAllLists(elmnt) {
      var x = document.getElementsByClassName("autocomplete-items");
      for (var i = 0; i < x.length; i++) {
        if (elmnt != x[i] && elmnt != inp) {
          x[i].parentNode.removeChild(x[i]);
        }
      }
    }
    document.addEventListener("click", function (e) {
        closeAllLists(e.target);
    });
}
autocomplete(document.getElementById("song_name"), song_names);
</script>
""" # .format(df['track_name'].tolist())  # Pass your list of song names here

def main():
    return None

def update(name):
    return f"Welcome to Gradio, {name}!"

mood_recommender = gr.Interface(
                                fn=recommend_songs,
                                inputs= [gr.Dropdown(choices=["HAPPY", "SAD", "SCARED", "ANGRY"]),
                                        gr.Slider(minimum=1960 , maximum=2024, label="Year Filter Min", step=1, value=1960),
                                        gr.Slider(minimum=1960 , maximum=2024, label="Year Filter Max", step=1, value=2024),
                                        gr.Slider(minimum=0, maximum=240, label="Tempo Filter min", step=1, value=0),
                                        gr.Slider(minimum=0, maximum=240, label="Tempo Filter max", step=1, value=240),
                                        gr.CheckboxGroup(choices=df['artist_name'].unique().tolist()[1:100], label="Exclude artists"),
                                        gr.Radio(choices=[True, False],label="Randomization"),
                                        gr.Slider(minimum=1, maximum=20, label="No. of song recommendation", step=1),
                                        gr.Slider(minimum=0,maximum=1, label="deviation"),
                                        gr.Slider(minimum=0,maximum=100, label="Popularity filter min", step= 1, value=0),
                                        gr.Slider(minimum=0,maximum=100, label="Popularity filter max", step=1, value=100)
                                        ],
                                outputs=gr.Textbox(autofocus=True,
                                                   autoscroll=True,),
                                title="Recommed song on moods",
                                description="Achieving a remarkable 90% accuracy, our mood-based recommendation system curates personalized playlists from a database of 1.3 million songs spanning 1960 to 2024. Utilizing classical ML techniques such as PCA, SVD, and K-NN, our system tailors music selections to match your mood. Accessible to all via RESTful APIs, discovering the perfect soundtrack has never been more effortless.",
                                thumbnail="/Users/zac/Downloads/Moodify (3).png",
                                clear_btn="Clear"
                                )

similar_song = gr.Interface(fn=find_similar_songs,
                            inputs=[gr.Textbox(label="Song name"),
                                    gr.Textbox(label="Artist name"),
                                    gr.Slider(minimum=1 , maximum=200, label="No. of similar songs", step=1, value=1),
                                    gr.Dropdown(choices=["HAPPY", "SAD", "SCARED", "ANGRY"]),
                                    gr.Slider(minimum=1960 , maximum=2024, label="Year Filter Min", step=1, value=1960),
                                    gr.Slider(minimum=1960 , maximum=2024, label="Year Filter Max", step=1, value=2024),
                                    gr.Slider(minimum=0, maximum=240, label="Tempo Filter min", step=1, value=0),
                                    gr.Slider(minimum=0, maximum=240, label="Tempo Filter max", step=1, value=240),
                                    gr.CheckboxGroup(choices=df['artist_name'].unique().tolist()[1:100], label="Exclude artists"),
                                    gr.Radio(choices=[True, False],label="Randomization"),
                                    gr.Slider(minimum=0,maximum=1, label="deviation"),
                                    gr.Slider(minimum=0,maximum=100, label="Popularity filter min", step= 1, value=0),
                                    gr.Slider(minimum=0,maximum=100, label="Popularity filter max", step=1, value=100)],
                            outputs=gr.Textbox(autofocus=True, 
                                               autoscroll=True,
                                               ),
                            title="Find Similar songs")


clustering_song = gr.Interface(fn=cluster_songs,
                               inputs=[gr.Dropdown(choices=["HAPPY", "SAD", "SCARED", "ANGRY", "RANDOM"]),
                                       gr.Slider(minimum=1,maximum=100, label="No. of clusters/Playlist", step= 1, value=1),
                                       gr.Radio(choices=[True, False],label="Random seed value"),
                                       ],
                               outputs=gr.Textbox(),
                               )

search = gr.Interface(fn=search_songs,
                      inputs=[gr.Textbox(label="Song name"),
                              gr.Textbox(label="Artist name")],
                      outputs=gr.TextArea(label="Output"),
                      title="Search for songs/artists in database",
                      submit_btn="Search")

demo = gr.TabbedInterface(title="Moodify",interface_list=[mood_recommender, similar_song, clustering_song, search] ,tab_names=["Recommed song on moods","Recommend similar songs","Playlist creater", "Database search"])


demo.launch()