import os
from googleapiclient.discovery import build
import re

api_key = os.environ.get('API_KEY')


# Initialize the YouTube Data API
youtube = build('youtube', 'v3', developerKey=api_key)

def extract_playlist_id(playlist_url):
    # Regular expression to extract playlist ID from URL
    pattern = r"list=([a-zA-Z0-9_-]+)"
    match = re.search(pattern, playlist_url)
    if match:
        return match.group(1)
    else:
        return None

def list_songs_in_playlist(playlist_url):
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

    # Extract song titles
    for item in playlist_items['items']:
        song_title = item['snippet']['title']
        songs.append(song_title)

    return songs



# Example usage
playlist_url = ["https://www.youtube.com/watch?v=Jkj36B1YuDU&list=PL3-sRm8xAzY-w9GS19pLXMyFRTuJcuUjy", "asdasdasd"]
for playlist in playlist_url:
    songs = list_songs_in_playlist(playlist_url)
    if songs:
        print("Songs in the playlist:")
        for song in songs:
            print(song)
