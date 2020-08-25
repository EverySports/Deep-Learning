from selenium import webdriver

search_term = f'lunge'
url = f'https://www.google.com/search?q={search_term}&tbm=isch'

browser = webdriver.Chrome(f'./chromedriver')
browser.get(url)

for i in range(200):
    browser.execute_script(f'window.scrollBy(0,10000)')

for idx, el in enumerate(browser.find_elements_by_class_name(f'rg_i')):
    el.screenshot(f'./lunge/{idx}.png')