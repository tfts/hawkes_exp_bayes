'''
This script compiles a dataset of word association to language level from http://www.englishprofile.org/american-english.
Date: 2020-03-13
Chrome (Chromium) Version: 79.0.3945.79, built and running on Ubuntu 19.04 
'''
import os
import time

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

BASE_URL = 'http://www.englishprofile.org/american-english'
CELL_TO_COL = {0: 'word', 1: 'guide', 2: 'level', 3: 'pos', 4: 'topic'}

df = {v: [] for v in CELL_TO_COL.values()}
driver = webdriver.Chrome()
driver.get(BASE_URL)
driver.find_element_by_id('filter_search').send_keys('*' + Keys.RETURN)
for page_i in range(0, 15390, 20):
    rows = driver.find_element_by_id('reportList').find_elements_by_tag_name('tr')
    for row in rows[1:]:
        for cell_i, cell in enumerate(row.find_elements_by_tag_name('td')):
            if cell_i == 5:
                continue
            elif cell_i == 0:
                print(cell.text)
            df[CELL_TO_COL[cell_i]].append(cell.text if cell.text else None)
    if page_i % 100 == 0:
        pd.DataFrame(df).to_csv('cerf_en.csv', index=None)
    time.sleep(3)
    driver.find_element_by_class_name('pagination-next').find_element_by_tag_name('a').click()
    driver.find_element_by_id('adminForm').find_element_by_tag_name('button').click()

pd.DataFrame(df).to_csv('cerf_en.csv', index=None)
