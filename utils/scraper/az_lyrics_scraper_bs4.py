from bs4 import BeautifulSoup
import requests

# URL of the page containing the lyrics
url = "https://www.azlyrics.com/lyrics/linkinpark/intheend.html"

# Fetch the page content
response = requests.get(url)
html_content = response.text

# Parse the HTML
soup = BeautifulSoup(html_content, "html.parser")

# Find the div containing the lyrics
# lyrics_div = soup.find("div", class_="col-xs-12 col-lg-8 text-center")
# # Extract the lyrics text
# lyrics = lyrics_div.text.strip()
# print(lyrics)

divs_without_class = soup.find_all('div', class_=False)

# Extract the text from these <div> elements
lyrics = ""
for div in divs_without_class:
    lyrics += div.get_text()

print(lyrics)


