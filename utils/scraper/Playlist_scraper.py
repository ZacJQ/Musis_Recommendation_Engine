from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time 
import os

# Sensational Soundscape: sensation : numbness

no_playlist = 15
output_list = []
error_list = []
j = 1
k = 0
error = 0
version = 6
part = 6
# os.chdir("/Users/zac/Codes/Music_Project/GIT_HUB/Musis_Recommendation_Engine")
root = os.getcwd()
logs_file_path = f"data/raw_data/scraped_data/playlist_scrapping_logs_v{version}.txt'"
remaining_file_path = "data/raw_data/scraped_data/error/remaining.csv"
output_file_path = f"data/raw_data/scraped_data/playlist_terms_part{part}.csv"
IS_FIRST_RUN = False


start_time = time.time()
def scroll_down():
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    WebDriverWait(driver, 60).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, "yt-formatted-string#message")))
    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-playlist-renderer")))



if IS_FIRST_RUN:
    gpt_list = """1. Chill Vibes: relaxation : stress
    2. Uplifting Anthems: happiness : sadness
    3. Melancholic Melodies: sadness : melancholy
    4. Power Ballads: empowerment : low self-esteem
    5. Jazzy Nights: nostalgia : loneliness
    6. Dancefloor Hits: excitement : boredom
    7. Rainy Day Tunes: introspection : gloominess
    8. Acoustic Serenade: comfort : heartache
    9. Inspirational Beats: motivation : discouragement
    10. Indie Gems: uniqueness : feeling unnoticed
    11. Feel-Good Favorites: positivity : negativity
    12. Summer Breeze: warmth : coldness
    13. Heartfelt Ballads: emotion : numbness
    14. Road Trip Jams: adventure : monotony
    15. Soulful Grooves: depth : shallowness
    16. Electro Pop Fun: energy : lethargy
    17. Rhythmic Rapture: movement : stillness
    18. Healing Harmonies: healing : pain
    19. Mellow Moods: calmness : agitation
    20. Reggae Relaxation: tranquility : restlessness
    21. Classic Rock Revival: nostalgia : disconnection
    22. Folk Feelings: authenticity : falseness
    23. Romantic Rhythms: love : heartbreak
    24. Hip Hop Therapy: empowerment : frustration
    25. Ambient Escapes: peace : chaos
    26. Bluesy Resonance: introspection : despair
    27. Country Comfort: familiarity : estrangement
    28. Pop Perfection: catchiness : indifference
    29. Smooth Jazz Soiree: sophistication : rawness
    30. Motivational Mix: determination : doubt
    31. Electronic Dreams: imagination : stagnation
    32. Lyrical Exploration: storytelling : emptiness
    33. Latin Fiesta: celebration : sorrow
    34. Feel-Good Funk: groove : stiffness
    35. Piano Ponderings: reflection : distraction
    36. Worldly Wonders: diversity : monotony
    37. Ambient Bliss: serenity : restlessness
    38. Soulful Reflections: introspection : confusion
    39. Dreamy Delights: whimsy : seriousness
    40. Rock Revival: rebellion : conformity
    41. Gospel Glory: faith : doubt
    42. R&B Rejuvenation: sensuality : inhibition
    43. Jazz Journey: improvisation : rigidity
    44. Indie Introspection: authenticity : superficiality
    45. Classical Calm: elegance : clumsiness
    46. Electronic Enchantment: euphoria : depression
    47. Folklore Feels: storytelling : detachment
    48. Nostalgic Nights: reminiscence : detachment
    49. Eclectic Excursions: diversity : sameness
    50. Harmonic Hope: optimism : pessimism
    51. Laid-back Lounge: ease : tension
    52. Melodic Memories: nostalgia : forgetfulness
    53. Sentimental Serenade: sentimentality : apathy
    54. Symphony of Serenity: tranquility : agitation
    55. Acoustic Affection: intimacy : distance
    56. Disco Fever: joy : apathy
    57. Alternative Avenue: individuality : conformity
    58. Blissful Beats: euphoria : despondency
    59. Coffeehouse Crooners: intimacy : loneliness
    60. Ethereal Echoes: otherworldliness : mundanity
    61. Groovy Goodness: rhythm : discord
    62. Harmonious Heights: unity : division
    63. Vibrant Voices: expression : repression
    64. Retro Rewind: nostalgia : detachment
    65. Sentimental Sojourn: sentimentality : indifference
    66. Ambient Adventure: exploration : stagnation
    67. Ballad Bouquet: emotion : numbness
    68. Dynamic Dreamscape: imagination : reality
    69. Enchanted Emotions: wonder : skepticism
    70. Jazzed-Up Jams: improvisation : predictability
    71. Lively Landscapes: energy : lethargy
    72. Poetic Portraits: lyricism : literalism
    73. Psychedelic Symphony: exploration : conventionality
    74. Radiant Rhythms: vibrancy : dullness
    75. Sensational Soundscape: sensation : numbness
    76. Serene Sonnets: tranquility : turmoil
    77. Spiritual Solace: connection : isolation
    78. Sultry Sensations: sensuality : inhibition
    79. Tender Tunes: gentleness : harshness
    80. Tranquil Trails: peace : unrest
    81. Upbeat Utopia: positivity : negativity
    82. Whimsical Wanderlust: imagination : reality
    83. Zen Zone: mindfulness : distraction
    84. Ambient Atmosphere: serenity : chaos
    85. Ballroom Bliss: elegance : awkwardness
    86. Cosmic Connections: universality : individualism
    87. Dreamy Disposition: whimsy : practicality
    88. Groove Garden: rhythm : dissonance
    89. Harmony Haven: unity : discord
    90. Jazzy Junction: improvisation : rigidity
    91. Luminescent Lullabies: brightness : darkness
    92. Melodic Meadows: tranquility : turbulence
    93. Peaceful Paradise: serenity : unrest
    94. Rhythmic Reverie: movement : stillness
    95. Sonic Sanctuary: refuge : exposure
    96. Tranquil Tempos: calmness : agitation
    97. Velvet Voices: intimacy : distance
    98. Whispering Winds: softness : harshness
    99. Zenith Zone: elevation : stagnation
    100. Ethereal Embrace: otherworldliness : mundanity"""

    search_terms = gpt_list.split("\n")
    final_list = []

    for terms in search_terms:
        final = terms.split(".")[-1]
        final_list.append(final.strip())
    remaining_list = final_list
else:


    final_list_df = pd.read_csv(os.path.join(root , remaining_file_path))
    final_list_df = final_list_df["0"]
    final_list = final_list_df.to_list()
    remaining_list = final_list


# final_list = [ "angry", "sad" , ""]

for line_item in final_list:
    remaining_list.pop(0)
    try:    
        # Start a new instance of Chrome web browser
        terms = line_item.split(':')[0]
        terms = terms + " songs"
        driver = webdriver.Chrome()
        driver.maximize_window()
        driver.get("https://www.youtube.com")
        search_bar = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.NAME, "search_query")))
        search_bar.send_keys(terms)
        search_button = driver.find_element(By.ID, "search-icon-legacy")
        search_button.click()
        time.sleep(1)
        filter_button = WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.ID, "filter-button")))
        filter_button.click()
        time.sleep(1)
        type_dropdown = WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.XPATH, "//div[@title='Search for Playlist']")))
        type_dropdown.click()
        time.sleep(1)
        

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
            link = playlist.find_element(By.XPATH, ".//a[@id='thumbnail']").get_attribute("href")
            print("_"*110)
            playlist_info.append({"name": name, "link": link})
            print(f"{i+1}. Name: {name}\n   Link: {link}\n")
            i+=1
            temp = {"search_no": j, "search_term":terms, "name_playlist": name, "link_playlist": link, "mood_mapping":line_item, "playlist_no": i }
            # output_dict.update(temp)
            output_list.append(temp)
            output_df = pd.DataFrame(output_list)
            output_df.to_csv(os.path.join(root, output_file_path), index=False)
        j = j + 1
        k = k+1
        WebDriverWait(driver, 10)
        driver.quit()
        remaining_df = pd.DataFrame(remaining_list)
        remaining_df.to_csv(os.path.join(root, remaining_file_path), index=False)
    except Exception as e:
        driver.quit()
        print(f"Faced an error {e}")
        output_df.to_csv(os.path.join(root, f"data/raw_data/scraped_data/error/playlist_{j}_{i}_terms_error_v{version}.csv"), index=False)
        error = error + 1 
        error_list.append({"error_no": error, "search_term":terms, "mood_mapping": line_item})
        remaining_df = pd.DataFrame(remaining_list)
        remaining_df.to_csv(os.path.join(root, remaining_file_path), index=False)
        stop_time = time.time()

        total_time = stop_time-start_time

        print(f"Total terms scraped - {j} \nTotal playlists scraped - {k}\nTotal errors - {error}\nTerms remaining - {len(remaining_list)} \nTotal time taken - {stop_time-start_time}")
        with open(os.path.join(root, logs_file_path), 'w') as f:
            f.write(f"Total terms scraped - {j}\n")
            f.write(f"Total playlists scraped - {k}\n")
            f.write(f"Total errors - {error}\n")
            f.write(f"Terms remaining - {len(remaining_list)}\n")
            f.write(f"Total time taken (in sec) - {total_time}\n")
            f.write(f"Time (in minutes) - {total_time/60}")
            f.write(f"Version - 1")


stop_time = time.time()
total_time = stop_time-start_time

print(f"Total terms scraped - {j} \nTotal playlists scraped - {k}\nTotal errors - {error}\nTerms remaining - {len(remaining_list)} \nTotal time taken - {stop_time-start_time}")


with open(os.path.join(root, logs_file_path), 'w') as f:
    f.write(f"Total terms scraped - {j}\n")
    f.write(f"Total playlists scraped - {k}\n")
    f.write(f"Total errors - {error}\n")
    f.write(f"Terms remaining - {len(remaining_list)}\n")
    f.write(f"Total time taken (in sec) - {total_time}\n")
    f.write(f"Time (in minutes) - {total_time/60}\n")
    f.write(f"Version - {version}")




