import selenium
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

def fetchLinks():
    options = webdriver.ChromeOptions()
    #options.add_argument('profile-directory=Private')
    options.add_argument('--headless')
    #driver = webdriver.Chrome(options=options)
    driver = webdriver.Chrome(ChromeDriverManager().install(), options = options)

    username = "CbjOpto"
    password = "RA5!CyqwPFi?G9N"

    # Open pavlovia
    driver.get("https://pavlovia.org")

    # Sign in
    signin = driver.find_element_by_xpath("/html/body/app-root/div/section/div/nav/div[2]/button[5]").click()

    #username
    user = driver.find_element_by_id("user_login").send_keys(username)

    #password
    webpass = driver.find_element_by_id("user_password").send_keys(password)

    #sign in button
    button = driver.find_element_by_name("commit").click()

    # Fetch Resting State Link
    url = "https://pavlovia.org/CbjOpto/restingstateeeg"
    driver.get(url)
    for i in range(100):
        while True:
            try:
                driver.find_element_by_xpath("/html/body/app-root/div/div/app-projects/section/div/div[2]/div[1]/div[1]/div[3]/div[2]/button").click()
                break
            except:
                continue
        break

    driver.switch_to.window(driver.window_handles[1])

    while True:
        if len(driver.current_url) > 50:
            RestUrl = driver.current_url
            print("Link1 fetched")
            break

    # Fetch N-Back1 Link
    url = "https://pavlovia.org/CbjOpto/nbacknewest"
    driver.get(url)
    driver.switch_to.window(driver.window_handles[1])
    for i in range(100):
        while True:
            try:
                driver.find_element_by_xpath("/html/body/app-root/div/div/app-projects/section/div/div[2]/div[1]/div[1]/div[3]/div[2]/button").click()
                break
            except:
                continue
        break

    driver.switch_to.window(driver.window_handles[2])

    while True:
        if len(driver.current_url) > 50:
            NBack1URL = driver.current_url
            print("Link2 fetched")
            break

    # Fetch N-Back2 Link
    url = "https://pavlovia.org/CbjOpto/test_nback_second"
    driver.get(url)
    driver.switch_to.window(driver.window_handles[2])
    for i in range(100):
        while True:
            try:
                driver.find_element_by_xpath("/html/body/app-root/div/div/app-projects/section/div/div[2]/div[1]/div[1]/div[3]/div[2]/button").click()
                break
            except:
                continue
        break

    driver.switch_to.window(driver.window_handles[3])

    while True:
        if len(driver.current_url) > 50:
            NBack2URL = driver.current_url
            print("Link3 fetched")
            break

    return RestUrl, NBack1URL, NBack2URL

#fetchLinks()