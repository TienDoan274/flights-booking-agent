from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time
from datetime import datetime, timedelta
from kafka import KafkaProducer
import json
producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
class WebCrawler:
    def __init__(self, url):
        self.url = url
        self.driver = None
        
    def initialize_driver(self):
        try:
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            return True
        except Exception as e:
            print(f"Failed to initialize driver: {str(e)}")
            return False

    def start_crawling(self):
        if not self.initialize_driver():
            return None
            
        try:
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.driver.get(self.url)
                    break
                except WebDriverException as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2)

            today = datetime.now()
            all_data = {}
            print(today)
            for i in range(5):  
                current_date = today + timedelta(days=i)
                date_str = current_date.strftime('%Y-%m-%d')
                print(f"\nProcessing date: {date_str}")

                date_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "flight_date"))
                )
                self.driver.execute_script(f"arguments[0].value = '{date_str}';", date_input)
                time.sleep(1)  
                search_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.btn-filter[type="submit"]'))
                )
                
                search_button.click()
                time.sleep(10)
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, 'tbody'))
                    )
                    
                    data = self.get_crawl_data()
                    for flight in data['data']:
                        information = {
                            'date':date_str,# ngày
                            'scheduled_time':flight[0],# Giờ kế hoạch ví dụ: 16:35
                            'updated_time':flight[1],# Giờ cập nhật ví dụ: 17:40
                            'route':flight[2],# Chặng bay ví dụ: DAD-HAN
                            'flight_id':flight[4],# Mã chuyến bay ví dụ: VJ528
                            'counter':flight[5],# Quầy ví dụ: 21-28
                            'gate':flight[6],# Cổng ví dụ: 7
                            'status':flight[8]# Trạng thái: OPN/CLS
                        }
                        producer.send('flights',value=information)
                    if data:
                        all_data[date_str] = data
                        print(data)
                        print(f"Successfully collected data for {date_str}")
                    
                except TimeoutException:
                    print(f"No data found for date {date_str}")
                    continue

                time.sleep(2)  

            return all_data
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
        finally:
            self.cleanup()
    
    def get_crawl_data(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print('Starting data extraction...')
                
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'table.table.table-striped'))
                )
                
                headers = []
                header_elements = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'table.table.table-striped thead tr th'))
                )
                for header in header_elements:
                    try:
                        headers.append(WebDriverWait(self.driver, 5).until(
                            EC.visibility_of(header)
                        ).text.strip())
                    except:
                        continue
                
                data_rows = []
                row_elements = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'table.table.table-striped tbody tr'))
                )
                
                for row in row_elements:
                    try:
                        cells = WebDriverWait(row, 5).until(
                            EC.presence_of_all_elements_located((By.TAG_NAME, 'td'))
                        )
                        
                        row_data = []
                        for cell in cells:
                            try:
                                cell_text = WebDriverWait(self.driver, 5).until(
                                    EC.visibility_of(cell)
                                ).text.strip()
                                row_data.append(cell_text)
                            except:
                                row_data.append("")  
                        if row_data and any(row_data):  
                            data_rows.append(row_data)
                            
                    except Exception as row_error:
                        print(f"Error processing row: {str(row_error)}")
                        continue
                
                if headers and data_rows:  
                    return {
                        'headers': headers,
                        'data': data_rows
                    }
                else:
                    raise Exception("No data found in table")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    print("Max retries reached, returning None")
                    return None
                time.sleep(2)  
            return None
    
    def cleanup(self):
        try:
            if self.driver:
                self.driver.quit()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

url = "https://vietnamairport.vn/thong-tin-lich-bay"
crawler = WebCrawler(url)
result = crawler.start_crawling()


    
