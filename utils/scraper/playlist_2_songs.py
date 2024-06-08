import os
from googleapiclient.discovery import build
import re
import pandas as pd

api_key = os.environ.get('YOUTUBE_API_KEY')

output = []
output_file = "data/raw_data/scraped_data/playlist_to_songs.csv"

# Initialize the YouTube Data API
youtube = build('youtube', 'v3', developerKey=api_key)

def extract_playlist_id(playlist_url: str)-> str | None:
    """
    
    """
    
    # Regular expression to extract playlist ID from URL
    pattern = r"list=([a-zA-Z0-9_-]+)"
    match = re.search(pattern, playlist_url)
    if match:
        return match.group(1)
    else:
        return None

def list_songs_in_playlist_old(playlist_url: str) -> tuple:
    """
    
    """
    playlist_id = extract_playlist_id(playlist_url)
    if not playlist_id:
        print("Invalid playlist URL.")
        return []

    songs = []

    # Retrieve playlist items
    playlist_items = youtube.playlistItems().list(
                                                playlistId=playlist_id,
                                                part='snippet',
                                                maxResults=50  # Adjust max results as per your requirement
                                                ).execute()

    playlist_response = youtube.playlists().list(
                                                part="snippet",
                                                id=playlist_id
                                                ).execute()

    # Extract the playlist name from the response
    playlist_name = playlist_response["items"][0]["snippet"]["title"]

    # Extract song titles
    for item in playlist_items['items']:
        song_title = item['snippet']['title']
        songs.append(song_title)

    return songs, playlist_name


def list_songs_in_playlist(playlist_url: str) -> tuple:
    """
    Lists the songs in a YouTube playlist along with their links.
    """
    playlist_id = extract_playlist_id(playlist_url)
    if not playlist_id:
        print("Invalid playlist URL.")
        return [], ""

    songs = []

    # Retrieve playlist items
    playlist_items = youtube.playlistItems().list(
                                                playlistId=playlist_id,
                                                part='snippet',
                                                maxResults=50  # Adjust max results as per your requirement
                                                ).execute()

    playlist_response = youtube.playlists().list(
                                                part="snippet",
                                                id=playlist_id
                                                ).execute()

    # Extract the playlist name from the response
    playlist_name = playlist_response["items"][0]["snippet"]["title"]

    # Extract song titles and links
    for item in playlist_items['items']:
        song_title = item['snippet']['title']
        video_id = item['snippet']['resourceId']['videoId']
        song_link = f"https://www.youtube.com/watch?v={video_id}"
        songs.append((song_title, song_link))

    return songs, playlist_name


root = os.getcwd()
relative_folder = "data/processed_data/scraped_data/playlist_terms_final_v1.csv"

playlist_df = pd.read_csv(os.path.join(root, relative_folder), index_col=False)

playlist_sr = playlist_df["link_playlist"]


playlist_url = playlist_sr.to_list()

# playlist_url = ["https://www.youtube.com/watch?v=Jkj36B1YuDU&list=PL3-sRm8xAzY-w9GS19pLXMyFRTuJcuUjy", "asdasdasd"]
print(len(playlist_url))


visited = []
for search in playlist_url:
    if search not in visited:
        visited.append(search)


print(len(visited))

for playlist in visited[13:17]:
    songs, playlist_name = list_songs_in_playlist(playlist)
    if songs:
        print("Songs in the playlist:")
        for song in songs:
            
            print(song)
            song_title, song_link = song
            print(playlist_name)
            print("END \n\n")

            temp = {"playlist_name": playlist_name , "youtube_playlist_link": playlist, "song_name": song_title , "song_link": song_link}
            output.append(temp)
            df = pd.DataFrame(output)
            df.to_csv(os.path.join(root, output_file))



