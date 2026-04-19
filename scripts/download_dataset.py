import os
from icrawler.builtin import BingImageCrawler

print("Starting download...")

landmarks = ["Taj Mahal", "Eiffel Tower", "India Gate", "Big Ben"]

base_path = "dataset/train"

for landmark in landmarks:
    folder_name = landmark.replace(" ", "_").lower()
    path = os.path.join(base_path, folder_name)
    os.makedirs(path, exist_ok=True)

    print(f"Downloading {landmark}...")

    crawler = BingImageCrawler(storage={'root_dir': path})
    crawler.crawl(keyword=landmark, max_num=40)

print("Dataset download complete!")