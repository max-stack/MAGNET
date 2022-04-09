from typing import Counter
from stop_words import get_stop_words
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from string import digits
from queue import PriorityQueue
import math


class TF_IDF:
    def __init__(self, processed_text):
        self.processed_text = processed_text
        self.DF = {}
        self.tf_idf = []

    def calculate_df(self):
        for i in range(len(self.processed_text)):
            words = self.processed_text[i].split()
            for word in words:
                try:
                    self.DF[word].add(i)
                except:
                    self.DF[word] = {i}

        for i in self.DF:
            self.DF[i] = len(self.DF[i])

    def calculate_tf_idf(self):
        self.calculate_df()
        for document in self.processed_text:
            tf_idf_list = []
            tf_idf_queue = PriorityQueue()
            for word in np.unique(document.split()):
                if word == "num":
                    continue
                tf = document.count(word)/len(document.split())
                df = self.DF[word]
                idf = math.log(len(self.processed_text)/(df+1))
                tf_idf = tf * idf
                if(tf_idf_queue.qsize() < 10):
                    tf_idf_queue.put([tf_idf, word])
                elif tf_idf > tf_idf_queue.queue[0][0]:
                    tf_idf_queue.get()
                    tf_idf_queue.put([tf_idf, word])
            while not tf_idf_queue.empty():
                tf_idf_list.append(tf_idf_queue.get()[1])
            self.tf_idf.append(tf_idf_list)


class PreprocessText:
    def __init__(self, original_text):
        self.original_text = original_text

    def preprocess(self):
        self.remove_numbers()
        self.convert_lower_case()
        self.remove_punctuation()
        self.remove_stop_words()
        self.remove_apostrophe()
        self.remove_single_chars()
        self.stemming()

    def remove_numbers(self):
        remove_digits = str.maketrans('', '', digits)
        for i in range(len(self.original_text)):
            self.original_text[i] = self.original_text[i].translate(
                remove_digits)

    def convert_lower_case(self):
        # lower case
        self.original_text = np.char.lower(self.original_text)

    def remove_punctuation(self):
        # remove punctuation
        symbols = "!\"#$%&()*+-.,/:;<=>?@[\]^_`{|}~\n"
        for symbol in symbols:
            self.original_text = np.char.replace(
                self.original_text, symbol, '')

    def remove_stop_words(self):
        # remove stop words
        for i in range(len(self.original_text)):
            new_text = ""
            words = self.original_text[i].split()
            for word in words:
                if word not in get_stop_words('en'):
                    new_text += word + " "
            self.original_text[i] = new_text

    def remove_apostrophe(self):
        # remove apostrophe
        self.original_text = np.char.replace(self.original_text, "'", "")

    def remove_single_chars(self):
        # remove single characters
        for i in range(len(self.original_text)):
            new_text = ""
            words = self.original_text[i].split()
            for word in words:
                if len(word) > 1:
                    new_text += word + " "
            self.original_text[i] = new_text

    def stemming(self):
        # stemming
        lemmatizer = WordNetLemmatizer()
        for i in range(len(self.original_text)):
            new_text = ""
            words = self.original_text[i].split()
            for word in words:
                new_text += lemmatizer.lemmatize(word) + " "
            self.original_text[i] = new_text
