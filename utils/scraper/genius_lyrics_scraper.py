from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep

artist_name = "linkin-park"
song_name = "in-the-end"
base_url = "https://genius.com"

chrome_options = Options()

# chrome_options.add_argument("--headless")

driver = webdriver.Chrome(options=chrome_options)

driver.get(url=f'{base_url}/{artist_name}-{song_name}-lyrics')
# sleep(10)

container_of_lyrics = driver.find_elements(By.CLASS_NAME, "Lyrics__Container-sc-1ynbvzw-1 kUgSbL")


for container in container_of_lyrics:
    lyrics = container.text
    print(lyrics)
# sleep(10)

driver.quit()
