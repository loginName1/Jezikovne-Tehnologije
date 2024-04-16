import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import requests
from bs4 import BeautifulSoup
import csv
import gzip
import json
import matplotlib


#### URL SCRAPING (PHISHTANK DISABLED NEW USER REGISTRATION)
class PhishTankUrlScraper(object):
    def __init__(self):
        self.url_home = 'https://www.phishtank.com/'
        self.url_search = self.url_home + 'phish_search.php?page={:d}&active=y&verified=y'
        self.phish_url_dict = {}

    @staticmethod
    def check_if_page_up_or_down(url=None):
        # to make the scraper work properly, visit "https://phishtank.com/index.php" and check your cookie value and paste it into this here header
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            "Accept-Language": "en-US,en;q=0.5",
            'Alt-Used': 'phishtank.com',
            'Connection': 'keep-alive',
            'Cookie': 'cf_clearance=FIfX71hQCUrYgabgakWuq5bb8l7d5HvBw1hMqXoJBE4-1713180935-1.0.1.1-LohN0sMyaApvxdIUp_h2BLXabA_cdMxShm8qHgPcLpecH694oYZ4cMwxDaDVwasSV2f4tZqzV2d8LVUbsOMdlw; PHPSESSID=tpll9eildlocdp599fkatd3e45igurf7; __cf_bm=uhfRPhh0vR_5Qv967Mfo5eQS0kCgvCPQ.h2TAj3_z1k-1713180941-1.0.1.1-tlGJiTyNnSt2OmGa8vmJ7de3apL1x_kw.0M5kXnWZApZZVhTIaLHGE7qHRbVDl_sLeLCCylOEIl2Tf7FE9811A',
            'Host': 'phishtank.com',
            'Sec-Fetch-Dest': 'document',
            'Sec - Fetch - Mode': 'navigate',
            'Sec - Fetch - Site': 'none',
            'Sec - Fetch - User': '?1',
            "Upgrade-Insecure-Requests": "1",
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
        }
        try:
            page_request = requests.get(url=url, headers=headers)
            if page_request.status_code == 200:
                return page_request.text
            else:
                return None
        except requests.RequestException as e:
            print("Error fetching page:", e)
            return None

    @staticmethod
    def grab_phish_url(target_url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            page_response = PhishTankUrlScraper.check_if_page_up_or_down(target_url)
            if page_response:
                page = BeautifulSoup(page_response, 'html.parser')
                all_links = page.find_all('b')
                phish_urls = [link.string for link in all_links if link.string and 'http' in link.string]
                return phish_urls[0] if phish_urls else None
            else:
                return None
        except Exception as e:
            print("Error grabbing phish URL:", e)
            return None

    def grab_phish_ids(self, page_response):
        try:
            phish_ids = []
            page = BeautifulSoup(page_response, 'html.parser')
            all_links = page.find_all(name='a')
            for link in all_links:
                link_href = link.get('href')
                if link_href and "phish_detail.php?phish_id=" in link_href:
                    phish_ids.append(self.url_home + link_href)
            if not phish_ids:
                return None
            for number, phish_link in enumerate(phish_ids):
                phish_url = self.grab_phish_url(phish_link)
                if phish_url:
                    print(number + 1, ' - ', phish_link, ' - ', phish_url)
                    self.phish_url_dict[phish_link] = phish_url
        except Exception as e:
            print("Error grabbing phish IDs:", e)

    def scrape(self, page_range=10):
        try:
            for page_id in range(page_range):
                print('*_*_' * 24)
                target_url = self.url_search.format(page_id)
                page_response = self.check_if_page_up_or_down(target_url)
                if page_response:
                    self.grab_phish_ids(page_response)
                else:
                    continue
            print('*==*=' * 24)
            return self.phish_url_dict
        except Exception as e:
            print("Error during scraping:", e)

    def save_to_csv(self, file_name='phishing_urls.csv'):
        try:
            with open(file_name, 'w', newline='', encoding="utf-8") as file:
                writer = csv.writer(file, delimiter=';')
                for link, url in self.phish_url_dict.items():
                    writer.writerow([1, url])
            print(f"Data successfully saved to {file_name}")
        except Exception as e:
            print(f"Error saving data to {file_name}: {e}")

# PREPARE DATASETS
# EXTRACT COMMON CRAWL DATA
def extract_urls_from_cdx_file(file_path, limit=100):
    urls = []
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()  # Remove newline character
            if not line:
                continue
            try:
                parts = line.split(' ')
                url = parts[3][1:-2]  # Extract the URL and remove the quotation marks and trailing comma
                line_count += 1
                if line_count % 1000 == 0:
                    urls.append(url)
                if len(urls) >= limit:
                    break
            except Exception as e:
                print(f'Error processing line: {line}. Error: {e}')
                continue
    return urls

def write_urls_to_csv(urls, csv_file):
    with open(csv_file, 'w', newline='', ) as f:
        writer = csv.writer(f, delimiter=';')
        #writer.writerow(['URL'])
        for url in urls:
            writer.writerow([0, url])

############## CREATING THE FULL DATASET ##############
# with open('phishing_urls.csv', 'r', encoding='utf-8') as file1:
#     reader1 = csv.reader(file1, delimiter=';')
#     phishingDataset = list(reader1)
#
# # Open and read the second CSV file
# with open('cc-index-urls.csv', 'r', encoding='utf-8') as file2:
#     reader2 = csv.reader(file2, delimiter=';')
#     legitDataset = list(reader2)
#
# # Combine the data from the two CSV files
# merged_data = phishingDataset + legitDataset
#
# # Write the merged data to a new CSV file
# with open('dataset.csv', 'w', newline='', encoding='utf-8') as merged_file:
#     writer = csv.writer(merged_file, delimiter=';')
#     writer.writerow(['Label', 'Url'])
#     writer.writerows(merged_data)

# specify GPU


device = torch.device("cuda")
df = pd.read_csv("dataset.csv", delimiter=';', on_bad_lines='warn', quotechar=';', encoding='utf-8', usecols=['Label', 'Url'])
df.head()


train_text, temp_text, train_labels, temp_labels = train_test_split(df['Url'], df['Label'],
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=df['Label'])


val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

# BERT IMPORTS AND PREP
bertBase = AutoModel.from_pretrained('google-bert/bert-base-uncased')
bertBaseC = AutoModel.from_pretrained('google-bert/bert-base-cased')
bertLarge = AutoModel.from_pretrained('google-bert/bert-large-uncased')

tokenizerBase = BertTokenizerFast.from_pretrained('google-bert/bert-base-uncased')
tokenizerBaseC = BertTokenizerFast.from_pretrained('google-bert/bert-base-cased')
tokenizerLarge = BertTokenizerFast.from_pretrained('google-bert/bert-large-uncased')

################ TOKENIZE BASE MODEL ################
tokens_train_base = tokenizerBase.batch_encode_plus(
    train_text.tolist(),
    padding='max_length',
    truncation=True
)
# tokenize and encode sequences in the validation set
tokens_val_base = tokenizerBase.batch_encode_plus(
    val_text.tolist(),
    padding='max_length',
    truncation=True
)
# tokenize and encode sequences in the test set
tokens_test_base = tokenizerBase.batch_encode_plus(
    test_text.tolist(),
    padding='max_length',
    truncation=True
)

################ TOKENIZE BASE CASED MODEL ################
tokens_train_basec = tokenizerBaseC.batch_encode_plus(
    train_text.tolist(),
    padding='max_length',
    truncation=True
)
# tokenize and encode sequences in the validation set
tokens_val_basec = tokenizerBaseC.batch_encode_plus(
    val_text.tolist(),
    padding='max_length',
    truncation=True
)
# tokenize and encode sequences in the test set
tokens_test_basec = tokenizerBaseC.batch_encode_plus(
    test_text.tolist(),
    padding='max_length',
    truncation=True
)

################ TOKENIZE LARGE MODEL ################
tokens_train_large = tokenizerLarge.batch_encode_plus(
    train_text.tolist(),
    padding='max_length',
    truncation=True
)
# tokenize and encode sequences in the validation set
tokens_val_large = tokenizerLarge.batch_encode_plus(
    val_text.tolist(),
    padding='max_length',
    truncation=True
)
# tokenize and encode sequences in the test set
tokens_test_large = tokenizerLarge.batch_encode_plus(
    test_text.tolist(),
    padding='max_length',
    truncation=True
)


############### BASE MODEL TEST ###############
train_seq = torch.tensor(tokens_train_base['input_ids'])
train_mask = torch.tensor(tokens_train_base['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val_base['input_ids'])
val_mask = torch.tensor(tokens_val_base['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test_base['input_ids'])
test_mask = torch.tensor(tokens_test_base['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

batch_size = 32
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)
# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)
# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

for param in bertBase.parameters():
    param.requires_grad = False



#if __name__ == '__main__':

    #################### Scraping code (DONT RUN IF POTATO PC) ####################
    #scraper = PhishTankUrlScraper()
    #phish_tank = scraper.scrape(page_range=600)
    #print("List of phish urls(total grabbed - {:d})".format(len(phish_tank)))
    #scraper.save_to_csv()

    ################## Common Crawl code (DONT RUN IF POTATO PC) ##################
    #file_path = 'cdx-00010'
    #csv_file = 'cc-index-urls.csv'
    #limit = 11769
    #urls = extract_urls_from_cdx_file(file_path, limit)
    #write_urls_to_csv(urls, csv_file)
    #print(f'Extracted {len(urls)} URLs and written to {csv_file}')
