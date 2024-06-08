from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep

artist_name = "linkinpark"
song_name = "intheend"
base_url = "https://www.azlyrics.com/lyrics"
# https://www.azlyrics.com/lyrics/linkinpark/intheend.html
chrome_options = Options()

# chrome_options.add_argument("--headless")

driver = webdriver.Chrome(options=chrome_options)

driver.get(url=f'{base_url}/{artist_name}/{song_name}.html')
# sleep(10)

container_of_lyrics = driver.find_elements(By.XPATH, "//div[@class='col-xs-12 col-lg-8 text-center']/div").text
for containers in container_of_lyrics:
    print(containers)

# print(container_of_lyrics)

driver.quit()
