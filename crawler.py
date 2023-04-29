from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse, urljoin
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from urllib.parse import urlparse
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
from typing import Tuple, List
from collections import deque
from sqlite3 import Error
from queue import Queue
import threading
import requests
import sqlite3
import logging
import string
import nltk
import time 
import re

class Crawler:
    def __init__(self):
        self.visited_urls = set()
        self.index = {}
        self.conn = sqlite3.connect('crawler.db')
        self.cursor = self.conn.cursor()
        self.sid = SentimentIntensityAnalyzer()

        # # Create a connection pool with a maximum of 5 connections
        # self.connection_pool = Queue(maxsize=5)

        self.url_queue = deque()

        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('english'))

        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # Add a console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # Define the log message format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                content TEXT,
                sentiment REAL DEFAULT 0.0
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS word_occurrences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER,
                word_id INTEGER,
                FOREIGN KEY(page_id) REFERENCES pages(id),
                FOREIGN KEY(word_id) REFERENCES words(id)
            )
        ''')

        # Create the inverted index table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS inverted_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word_id INTEGER,
                page_id INTEGER,
                count INTEGER,
                FOREIGN KEY(word_id) REFERENCES words(id),
                FOREIGN KEY(page_id) REFERENCES pages(id)
            )
        ''')

        self.conn.commit()

    def create_connection(self):
        # connection = None
        # try:
        #     connection = self.connection_pool.get(timeout=1)
        # except:
        #     print("Connection pool exhausted")
        # return connection or sqlite3.connect('crawler.db')
        return sqlite3.connect('crawler.db')

    def release_connection(self,connection):
        connection.close()

    def _get_next_url(self):
        # Return the next URL in the queue, if there is one
        if self.url_queue:
            next_url = self.url_queue.popleft()
            return next_url
        # If there are no unvisited URLs, return None
        return None

    def _canonicalize_url(self, base_url, url):
        return urljoin(base_url, url)

    def _add_url_to_queue(self, base_url, url):
        # Convert the URL to a canonical form
        url = self._canonicalize_url(base_url, url)
        if url not in self.visited_urls and url not in self.url_queue:
            self.url_queue.append(url)

    def _get_word_id(self, word, cursor):
        cursor.execute('INSERT OR IGNORE INTO words (word) VALUES (?)', (word,))
        cursor.execute('SELECT id FROM words WHERE word = ?', (word,))
        return cursor.fetchone()[0]

    def _get_words(self, text):
        """Extracts all words from a text string, removing punctuation and digits."""
        words = []
        for word in text.split():
            # Remove punctuation and digits
            word = re.sub(r'[^A-Za-z]', '', word)
            # Append the word to the list if it's not empty
            if word:
                words.append(word.lower())
        return words

    def _crawl_page(self, url):
        self.logger.info(f'Crawling {url}')
        try:
            # Create a new connection to the database for this worker thread
            conn = self.create_connection()
            cursor = conn.cursor()

            # Download the page content
            response = requests.get(url)

            # Extract the page content
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.get_text()

            # Get all text sentiment score
            scores = sid.polarity_scores(content)

            # Get all the links on the page
            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    self._add_url_to_queue(url, href)

            # Add the page to the database
            cursor.execute('INSERT OR IGNORE INTO pages (url, content, sentiment) VALUES (?, ?, ?)', (url, content, scores['compound']))
            page_id = cursor.lastrowid

            # Get the words on the page and add them to the index
            words = self._get_words(content)
            for word in words:
                # Skip stop words
                if word in self.stop_words:
                    continue
                word_id = self._get_word_id(word, cursor)
                cursor.execute('INSERT INTO word_occurrences (page_id, word_id) VALUES (?, ?)', (page_id, word_id))
                # Check if the inverted index already contains an entry for this word/page combination
                cursor.execute('SELECT id FROM inverted_index WHERE word_id = ? AND page_id = ?', (word_id, page_id))
                result = cursor.fetchone()
                if result:
                    inverted_index_id = result[0]
                    cursor.execute('UPDATE inverted_index SET count = count + 1 WHERE id = ?', (inverted_index_id,))
                else:
                    cursor.execute('INSERT INTO inverted_index (word_id, page_id, count) VALUES (?, ?, 1)', (word_id, page_id))

            conn.commit()
            self.logger.info(f'Crawled {url}')
        except Exception as e:
            self.logger.exception(f'Error crawling {url}: {e}')
        finally:
            if conn:
                self.release_connection(conn)

    def _preprocess(self,text):
        # Lowercase the text
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove digits
        text = re.sub(r"\d+", "", text)

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stop words
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

        # Stem the words
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(word) for word in tokens]

        return tokens

    def crawl(self, seed_url):
        self.url_queue.append(seed_url)
        self.logger.info(f'Crawling {seed_url}')
        while True:
            # Get the next URL to crawl
            url = self._get_next_url()
            print(url)
            if not url:
                # If there are no more URLs to crawl, stop
                break
            # Crawl the page
            self._crawl_page(url)

    """ SEARCH WITH INVERTED INDEX """
    def search(self,query, threshold=None):
        words = self._preprocess(query)
        self.logger.info(f'Searching for {query}')
        conn = self.create_connection()
        cursor = conn.cursor()

        # Get the page IDs for all pages that contain any of the words
        cursor.execute('SELECT DISTINCT page_id FROM inverted_index WHERE word_id IN (SELECT id FROM words WHERE word IN (%s))' % ','.join(['?' for _ in words]), words)
        results = cursor.fetchall()

        page_counts = {}
        for result in results:
            page_id = result[0]
            count = 0
            for word in words:
                cursor.execute('SELECT count FROM inverted_index WHERE word_id = (SELECT id FROM words WHERE word = ?) AND page_id = ?', (word, page_id))
                result = cursor.fetchone()
                if result is not None:
                    count += result[0]
            page_counts[page_id] = count

        # Sort the pages by the number of occurrences of the search words in each page
        sorted_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)

        # Get the URLs for the top 10 pages and return them
        urls = []
        for page_id, count in sorted_pages[:10]:
            cursor.execute('SELECT url FROM pages WHERE id = ?', (page_id,))
            result = cursor.fetchone()
            if result is not None:
                urls.append(result[0])
        conn.close()

        return urls[0]

    def wn_search(self,query):
        # Preprocess query
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w.lower()) for w in query.split()]

        self.logger.info(f'Searching WN for {query}')

        # Find similar words using WordNet
        synsets = []
        for word in words:
            synset = wordnet.synsets(word)
            if synset:
                synsets.append(synset)
        similar_words = set()
        for synset in synsets:
            for lemma in synset[0].lemmas():
                similar_word = stemmer.stem(lemma.name().lower())
                if similar_word not in words:
                    similar_words.add(similar_word)
        v = ' '.join(list(similar_words))

        self.logger.info(f'WN words: {v}')

        # Add similar words to search words
        words = list(similar_words)

        conn = self.create_connection()
        cursor = conn.cursor()

        # Get the page IDs for all pages that contain any of the words
        cursor.execute('SELECT DISTINCT page_id FROM inverted_index WHERE word_id IN (SELECT id FROM words WHERE word IN (%s))' % ','.join(['?' for _ in words]), words)
        results = cursor.fetchall()

        page_counts = {}
        for result in results:
            page_id = result[0]
            count = 0
            for word in words:
                cursor.execute('SELECT count FROM inverted_index WHERE word_id = (SELECT id FROM words WHERE word = ?) AND page_id = ?', (word, page_id))
                result = cursor.fetchone()
                if result is not None:
                    count += result[0]
            page_counts[page_id] = count

        # Sort the pages by the number of occurrences of all search words in each page
        sorted_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)

        # Get the URLs for the top 10 pages and return them
        urls = []
        for page_id, count in sorted_pages[:10]:
            cursor.execute('SELECT url FROM pages WHERE id = ?', (page_id,))
            result = cursor.fetchone()
            if result is not None:
                urls.append(result[0])
        conn.close()

        return urls[0]