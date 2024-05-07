from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scroll_down():
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    WebDriverWait(driver, 60).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, "yt-formatted-string#message")))
    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-playlist-renderer")))

search_terms = ["angry songs" , "sad songs", "chill songs"]
no_playlist = 15



for terms in search_terms:
    # Start a new instance of Chrome web browser
    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get("https://www.youtube.com")
    search_bar = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.NAME, "search_query")))
    search_bar.send_keys(terms)
    search_button = driver.find_element(By.ID, "search-icon-legacy")
    search_button.click()
    filter_button = WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.ID, "filter-button")))
    filter_button.click()
    type_dropdown = WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.XPATH, "//div[@title='Search for Playlist']")))
    type_dropdown.click()

    # Scroll down multiple times to load more playlists
    for _ in range(7):  # Adjust the number of times to scroll as needed
        scroll_down()

    
    WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, "//ytd-playlist-renderer")))
    playlist_elements = driver.find_elements(By.XPATH, "//ytd-playlist-renderer")

    while len(playlist_elements) < no_playlist:
        scroll_down()
        playlist_elements = driver.find_elements(By.XPATH, "//ytd-playlist-renderer")

    playlist_elements = driver.find_elements(By.XPATH, "//ytd-playlist-renderer")

    class_name = "style-scope ytd-item-section-renderer"
    playlist_info = []

    i = 0
    for playlist in playlist_elements:
        if i >= no_playlist:
            break
        print("_"*110)
        name = playlist.find_element(By.XPATH, ".//span[@id='video-title']").text
        link = playlist.find_element(By.XPATH, "//a[@class='yt-simple-endpoint style-scope ytd-playlist-renderer']").get_attribute("href")
        print("_"*110)
        playlist_info.append({"name": name, "link": link})
        print(f"{i+1}. Name: {name}\n   Link: {link}\n")
        i+=1
    WebDriverWait(driver, 10)
    driver.quit()





