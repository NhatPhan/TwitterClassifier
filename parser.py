import string

import os
from os import listdir
from os.path import isfile, join
from os import walk
import csv
import json


class Parser():
    
    def get_clean_text(self, file_name):
        texts = []
        with open(file_name,"rb") as f:
            for line in f:
                while True:
                    try:
                        jfile = json.loads(line)
                        text = jfile['text']
                        text = text.encode('utf-8')
                        text.strip()
                        text = ''.join(s for s in text if s.isalnum() or s in string.whitespace)
                        text = text.replace('\n', ' ').replace('\r', '').replace('\t', '')
                        texts.append(text)
                        break
                    except ValueError:
                        # Not yet a complete JSON value
                        line += next(f)
        return texts;
    
    def parse_training_docs(self, source="training_tweets", dest="training_tweets/categorized_tweets.csv"):
        """
        write to dest file with headers : "text","category",
        category is the folder name under source and text is the
        file name under category folder (files converted to text)
        """
        # Get all categories
        categorized_folders = []
        for (dirpath, dirnames, filenames) in walk(source):
            categorized_folders.extend(dirnames)
            break    

        # Get (text, category) tuples
        categorized_files = []
        for folder in categorized_folders:
            folder_path = source + "/" + folder
            files = [(folder_path + "/" + file_name, folder) for file_name in listdir(folder_path) if isfile(join(folder_path, file_name))]
            categorized_files.extend(files)
    
        text_category_dict = {}
        
        for file_name, category in categorized_files:
            # extract text and write to file
            texts = self.get_clean_text(file_name)
            for text in texts:
                text_category_dict[text] = category

    
        with open(dest, 'wb') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = ["Text", "Category"])
            writer.writeheader()
            writer = csv.writer(csv_file, lineterminator='\n')
            for key, value in text_category_dict.items():
                writer.writerow([key, value])
            

parser = Parser()
parser.parse_training_docs()

