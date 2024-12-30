import os
import requests
from bs4 import BeautifulSoup
import re

# Hàm tạo thư mục lưu trữ
def create_save_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Hàm lưu nội dung vào tệp .txt
def save_content_to_file(content, filename, directory="crawled_data"):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)

# Hàm lấy tất cả các liên kết trên một trang
def get_links(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Lấy tất cả các liên kết từ thẻ <a> (URL hợp lệ)
            main_content_div = soup.find('div', class_='main-content')
            links = main_content_div.find_all('a', href=True)
            for i in links:
                if not i['href'].startswith(('https://')):
                    i['href'] =  'https://www.vietnamairlines.com/vn' + i['href'] 
            return [link['href'] for link in links if re.match(r'^https?://www.vietnamairlines.com/vn', link['href'])]
    except Exception as e:
        print(f"Lỗi khi lấy liên kết từ {url}: {e}")
    return []

# Hàm DFS để crawler theo chiều sâu và lưu nội dung
def dfs_crawler(url, max_depth, visited=None, depth=0, save_dir="crawled_data"):
    if visited is None:
        visited = set()

    if depth > max_depth or url in visited:
        return

    print(f"{'  ' * depth}Đang crawl: {url} (depth={depth})")
    visited.add(url)

    # Lấy nội dung từ URL hiện tại
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            if depth > 1:
                main_content_div = soup.find('div', class_='baggage-content')
            else: 
                main_content_div = soup.find('div', class_='main-content')
            content = "\n".join(tag.get_text().strip() for tag in main_content_div.find_all(['p', 'h1', 'h2', 'h3', 'ul', 'table']))
            if content:
                filename = f"url_depth_{depth}_{len(visited)}.txt"
                save_content_to_file(content, filename, save_dir)
    except Exception as e:
        print(f"{'  ' * depth}Lỗi khi lấy nội dung từ {url}: {e}")
        return

    # Lấy các liên kết trên trang
    links = get_links(url)

    # Duyệt qua các liên kết (theo DFS)
    for link in links:
        dfs_crawler(link, max_depth, visited, depth + 1, save_dir)

# URL bắt đầu
start_url = "https://www.vietnamairlines.com/vn/vi/travel-information/baggage/baggage-allowance-hand-baggage"  # Thay bằng URL gốc

# Thư mục lưu nội dung
save_directory = "crawled_data"

# Tạo thư mục lưu trữ nếu chưa tồn tại
create_save_dir(save_directory)

# Crawler theo chiều sâu với độ sâu tối đa là 2
dfs_crawler(start_url, max_depth=10, save_dir=save_directory)
