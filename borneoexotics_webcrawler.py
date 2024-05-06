import requests
import time
import os
from package.webcrawler_function import image_data_multilabel_prediction, rows_dict_to_csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

opts = Options()
opts.add_argument("--window-size=240,1440")

URL = "https://borneoexotics.net/shop/"
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
driver.get(URL)

species_group = ['truncata','veitchii','ventricosa']
folder_path = 'be_images'

def store_image_and_csv(folder_path, species_group):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    search_images_ul = driver.find_element(By.XPATH,'//*[@id="shop-isle-blog-container"]/ul[1]')
    search_images_li = search_images_ul.find_elements(By.TAG_NAME, 'li')
    print("len(search_images_li):",len(search_images_li))

    for images_class in search_images_li:
        ext = images_class.find_element(By.TAG_NAME, 'h2').text
        print(ext)
        if '–' not in ext:
            species = ext.split(':')[0].strip()  
            serial = ext.split('.')[-1].split(' ')[-1]              
            source = "unknown"    
        else:                  
            species = ext.split('–')[0].strip()                
            serial = ext.split('.')[-1].split(' ')[-1]
            source = ext.split('–')[-1].split(':')[0].strip()          
            
        images = images_class.find_elements(By.TAG_NAME, 'img')[0]
        IMAGE_URL = images.get_attribute('src')            
        headers = {'User-Agent': 'PostmanRuntime/7.29.0'}
        img_resource = requests.get(IMAGE_URL, headers=headers)
        if img_resource.status_code == 200:
            file_path = open(os.path.join(folder_path,f'{species}_{serial}_{source}.jpg'), 'wb')
            file_path.write(img_resource.content)
            file_path.close()
            print("file_path",file_path)
        else:
            print('status code error')  

    rows_dict = image_data_multilabel_prediction(folder_path, species_group=species_group, model_structure="model_structure.json", model_weights="model_weights.h5")

    rows_dict_to_csv(rows_dict,species_group=species_group, csvfile="images.csv")

    img_resource.close()
 
def img_webcrawler(folder_path, species_group):
    locator = (By.ID,"popmake-368")
    try:
        WebDriverWait(driver, 60).until(ec.visibility_of_element_located(locator))      
        driver.find_element(By.XPATH,'//*[@id="popmake-368"]/button').click()    
        time.sleep(2)
    except:
        pass

    for species in species_group:
        print("enter species" )
        search_species = driver.find_element(By.XPATH,'//*[@id="woocommerce-product-search-field-0"]')
        print("search_species:",search_species)
        search_species.clear()
        search_species.send_keys(species)
        search_species.send_keys(Keys.ENTER)

        time.sleep(1)
        
        try:
            page_div = driver.find_element(By.XPATH,'//*[@id="shop-isle-blog-container"]/div/div/nav')
            page_elements = page_div.find_elements(By.TAG_NAME, 'li')   
            for page in range(len(page_elements)-1):
                
                try:        
                    next_page = driver.find_element(By.CLASS_NAME, 'next') 
                    print("next_page:",next_page.text)
                    
                    next_page.click()                
                    store_image_and_csv(folder_path, species_group)
                except:
                    print("except")
                    pass
        except:
            pass  

    driver.quit()

if __name__ == '__main__':
    img_webcrawler(folder_path, species_group)