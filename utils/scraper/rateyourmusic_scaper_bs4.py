import requests
from bs4 import BeautifulSoup

artist_name = "linkin-park"
song_name = "in-the-end"
base_url = "https://rateyourmusic.com/release/single"

url = f"{base_url}/{artist_name}/{song_name}/"

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

album_info = soup.find(class_="album_info").get_text()
print(album_info)

cover_image = soup.find('div', class_='coverart').find('img')['src']
print(cover_image)
