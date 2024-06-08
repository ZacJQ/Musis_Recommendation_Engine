from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

artist_name = "linkin-park"
song_name = "in-the-end"
base_url = "https://rateyourmusic.com/release/single"

driver = webdriver.Chrome()
driver.get(url=f'{base_url}/{artist_name}/{song_name}/')



hey = driver.find_element(By.CLASS_NAME, "album_info" ).text 
print(hey)
link = driver.find_element(By.XPATH, '//div[contains(@class, "coverart")]/img').get_attribute("src")
print(link) 



driver.quit()