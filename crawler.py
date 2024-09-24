import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
import sqlite3
from collections import deque
import logging
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import BertTokenizer, BertModel
import torch
from scipy.sparse import csr_matrix
import warnings

# Suppression des avertissements FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

class Crawler:
    def __init__(self, starting_urls, max_urls=1000000, num_workers=5, delay=1, max_depth=5):
        self.starting_urls = starting_urls
        self.max_urls = max_urls
        self.num_workers = num_workers
        self.delay = delay
        self.max_depth = max_depth
        self.to_crawl = asyncio.Queue()
        self.mini_queue = deque(maxlen=1000)
        self.crawled_urls = set()
        self.domain_last_crawled = {}
        self.rp_cache = {}
        self.session = None
        self.db_conn = None
        self.stop_words = set(stopwords.words('english') + stopwords.words('french'))
        self.should_stop = asyncio.Event()
        
        # BERT model for semantic analysis
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # LDA model for page categorization
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        self.vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
        
        # PageRank
        self.pagerank = {}
        self.outgoing_links = {}

    async def init_db(self):
        try:
            self.db_conn = sqlite3.connect('crawler_data.db')
            logger.info("Connected to database: crawler_data.db")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

        cursor = self.db_conn.cursor()
        
    async def init_db(self):
        try:
            self.db_conn = sqlite3.connect('crawler_data.db')
            logger.info("Connected to database: crawler_data.db")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

        cursor = self.db_conn.cursor()
        
        # Définition des schémas de table
        table_schemas = {
            'pages': '''
                CREATE TABLE IF NOT EXISTS pages (
                    url TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    html_structure TEXT,
                    category INTEGER
                )
            ''',
            'queue': '''
                CREATE TABLE IF NOT EXISTS queue (
                    url TEXT PRIMARY KEY,
                    depth INTEGER
                )
            ''',
            'inverted_index': '''
                CREATE TABLE IF NOT EXISTS inverted_index (
                    word TEXT,
                    url TEXT,
                    tf_idf REAL,
                    semantic_vector BLOB,
                    PRIMARY KEY (word, url)
                )
            ''',
            'links': '''
                CREATE TABLE IF NOT EXISTS links (
                    from_url TEXT,
                    to_url TEXT,
                    PRIMARY KEY (from_url, to_url)
                )
            ''',
            'pagerank': '''
                CREATE TABLE IF NOT EXISTS pagerank (
                    url TEXT PRIMARY KEY,
                    score REAL
                )
            '''
        }

        # Création des tables si elles n'existent pas
        for table, schema in table_schemas.items():
            try:
                cursor.execute(schema)
                logger.info(f"Table '{table}' created or already exists.")
            except sqlite3.Error as e:
                logger.error(f"Error creating table '{table}': {str(e)}")

        # Vérifier et mettre à jour la structure de la table 'queue'
        try:
            cursor.execute("PRAGMA table_info(queue)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'depth' not in columns:
                cursor.execute("ALTER TABLE queue ADD COLUMN depth INTEGER")
                logger.info("Added 'depth' column to 'queue' table")
            self.db_conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating 'queue' table: {str(e)}")

        self.db_conn.commit()
        cursor.close()

    async def fetch(self, url, session):
        try:
            async with session.get(url, timeout=10, ssl=False, allow_redirects=True) as response:
                if response.status == 200:
                    final_url = str(response.url)
                    if final_url != url:
                        logger.info(f"Redirected from {url} to {final_url}")
                    return await response.text(), final_url
                else:
                    logger.warning(f"HTTP {response.status} for URL: {url}")
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching {url}: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
        return None, None

    def parse(self, html, url):
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.title.string if soup.title else ""
            content = soup.get_text()
            links = [urljoin(url, link.get('href')) for link in soup.find_all('a', href=True)]
            
            # Extract HTML structure
            html_structure = {
                'title': title,
                'h1': [h1.get_text() for h1 in soup.find_all('h1')],
                'h2': [h2.get_text() for h2 in soup.find_all('h2')],
                'h3': [h3.get_text() for h3 in soup.find_all('h3')],
                'p': [p.get_text() for p in soup.find_all('p')][:5]  # First 5 paragraphs
            }
            
            return title, content, links, html_structure
        except Exception as e:
            logger.error(f"Error parsing {url}: {str(e)}")
            return "", "", [], {}

    async def crawl_page(self, url, depth=0):
        if depth > self.max_depth:
            return

        if len(self.crawled_urls) >= self.max_urls:
            logger.info(f"Reached max URLs limit ({self.max_urls}). Stopping crawl.")
            self.should_stop.set()
            return

        domain = urlparse(url).netloc
        if domain in self.domain_last_crawled:
            time_since_last_crawl = time.time() - self.domain_last_crawled[domain]
            if time_since_last_crawl < self.delay:
                await asyncio.sleep(self.delay - time_since_last_crawl)

        if not self.can_fetch(url):
            return

        html, final_url = await self.fetch(url, self.session)
        if html:
            title, content, links, html_structure = self.parse(html, final_url)
            category = self.categorize_page(content)
            self.save_to_db(final_url, title, content, html_structure, category)
            self.crawled_urls.add(final_url)
            self.domain_last_crawled[domain] = time.time()

            for link in links:
                if is_valid_url(link) and link not in self.crawled_urls and len(self.crawled_urls) < self.max_urls:
                    await self.add_to_queue((link, depth + 1))
                self.add_link(final_url, link)

            await self.index_page(final_url, title, content, html_structure)

        logger.info(f"Crawled: {final_url}")

    def can_fetch(self, url):
        try:
            domain = urlparse(url).netloc
            if domain not in self.rp_cache:
                rp = RobotFileParser()
                rp.set_url(f"http://{domain}/robots.txt")
                rp.read()
                self.rp_cache[domain] = rp
            return self.rp_cache[domain].can_fetch("*", url)
        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {str(e)}")
            return True

    def save_to_db(self, url, title, content, html_structure, category):
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO pages (url, title, content, html_structure, category)
                VALUES (?, ?, ?, ?, ?)
            """, (url, title, content, str(html_structure), category))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")

    async def add_to_queue(self, url_depth):
        self.mini_queue.append(url_depth)
        if len(self.mini_queue) >= 100:
            await self.flush_queue_to_db()

    async def flush_queue_to_db(self):
        if not self.mini_queue:
            return
        try:
            cursor = self.db_conn.cursor()
            # Vérifier si la colonne 'depth' existe
            cursor.execute("PRAGMA table_info(queue)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'depth' in columns:
                cursor.executemany("INSERT OR IGNORE INTO queue (url, depth) VALUES (?, ?)",
                                self.mini_queue)
            else:
                # Si la colonne 'depth' n'existe pas, insérer seulement l'URL
                cursor.executemany("INSERT OR IGNORE INTO queue (url) VALUES (?)",
                                [(url,) for url, _ in self.mini_queue])
            self.db_conn.commit()
            self.mini_queue.clear()
        except Exception as e:
            logger.error(f"Error flushing queue to database: {str(e)}")

    async def get_next_url(self):
        if not self.mini_queue:
            try:
                cursor = self.db_conn.cursor()
                # Vérifier si la colonne 'depth' existe
                cursor.execute("PRAGMA table_info(queue)")
                columns = [col[1] for col in cursor.fetchall()]
                if 'depth' in columns:
                    cursor.execute("SELECT url, depth FROM queue LIMIT 1000")
                else:
                    cursor.execute("SELECT url FROM queue LIMIT 1000")
                urls = cursor.fetchall()
                if urls:
                    self.mini_queue.extend(urls if 'depth' in columns else [(url, 0) for (url,) in urls])
                    cursor.execute("DELETE FROM queue WHERE url IN (" + ",".join("?" * len(urls)) + ")",
                                [url[0] for url in urls])
                    self.db_conn.commit()
            except Exception as e:
                logger.error(f"Error getting next URL from database: {str(e)}")

        if self.mini_queue:
            return self.mini_queue.popleft()
        elif not self.to_crawl.empty():
            return await self.to_crawl.get(), 0
        else:
            return None, None


    async def index_page(self, url, title, content, html_structure):
        try:
            words = word_tokenize(f"{title} {content}")
            word_count = {}
            for word in words:
                word = word.lower()
                if word not in self.stop_words and word.isalnum():
                    word_count[word] = word_count.get(word, 0) + 1

            total_words = sum(word_count.values())
            
            cursor = self.db_conn.cursor()
            for word, count in word_count.items():
                tf = count / total_words
                cursor.execute("SELECT COUNT(*) FROM pages")
                total_docs = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM inverted_index WHERE word = ?", (word,))
                word_docs = cursor.fetchone()[0] + 1
                idf = math.log(total_docs / word_docs)
                tf_idf = tf * idf
                
                # Calculate semantic vector
                semantic_vector = self.get_semantic_vector(word)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO inverted_index (word, url, tf_idf, semantic_vector)
                    VALUES (?, ?, ?, ?)
                """, (word, url, tf_idf, semantic_vector.tobytes()))
            
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error indexing page {url}: {str(e)}")

    def get_semantic_vector(self, word):
        inputs = self.bert_tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def categorize_page(self, content):
        try:
            X = self.vectorizer.fit_transform([content])
            topic_distribution = self.lda_model.fit_transform(X)
            return topic_distribution.argmax()
        except ValueError as e:
            logger.warning(f"Error in categorize_page: {str(e)}. Returning default category 0.")
            return 0
        
    def save_pagerank_scores(self):
        try:
            cursor = self.db_conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO pagerank (url, score)
                VALUES (?, ?)
            """, self.pagerank.items())
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error saving PageRank scores: {str(e)}")

    def add_link(self, from_url, to_url):
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO links (from_url, to_url) VALUES (?, ?)", (from_url, to_url))
            self.db_conn.commit()
            
            if from_url not in self.outgoing_links:
                self.outgoing_links[from_url] = set()
            self.outgoing_links[from_url].add(to_url)
        except Exception as e:
            logger.error(f"Error adding link: {str(e)}")

    def calculate_pagerank(self, damping_factor=0.85, num_iterations=100):
        num_pages = len(self.crawled_urls)
        if num_pages == 0:
            return

        # Initialize PageRank
        for url in self.crawled_urls:
            self.pagerank[url] = 1 / num_pages

        for _ in range(num_iterations):
            new_ranks = {}
            for url in self.crawled_urls:
                rank = (1 - damping_factor) / num_pages
                if url in self.outgoing_links:
                    for incoming_url in self.outgoing_links[url]:
                        if incoming_url in self.pagerank:
                            num_outlinks = len(self.outgoing_links.get(incoming_url, []))
                            if num_outlinks > 0:
                                rank += damping_factor * (self.pagerank[incoming_url] / num_outlinks)
                new_ranks[url] = rank
            self.pagerank = new_ranks

    async def worker(self):
        while not self.should_stop.is_set():
            try:
                url, depth = await self.get_next_url()
                if url is None:
                    break
                if url not in self.crawled_urls and is_valid_url(url):
                    await self.crawl_page(url, depth)
            except Exception as e:
                logger.error(f"Error in worker: {str(e)}")

    async def stop_workers(self):
        self.should_stop.set()
        for _ in range(self.num_workers):
            await self.to_crawl.put((None, None))

    async def run(self):
        logger.info("Starting crawler...")
        await self.init_db()
        self.session = aiohttp.ClientSession()

        for url in self.starting_urls:
            await self.to_crawl.put(url)

        workers = [asyncio.create_task(self.worker()) for _ in range(self.num_workers)]

        try:
            await asyncio.wait_for(asyncio.gather(*workers), timeout=3600)  # 1 heure de timeout
        except asyncio.TimeoutError:
            logger.warning("Crawling timed out after 1 hour")
        except Exception as e:
            logger.error(f"Error during crawling: {str(e)}")
        finally:
            await self.stop_workers()
            await self.flush_queue_to_db()
            await self.session.close()
        
        logger.info("Calculating PageRank...")
        if self.crawled_urls:
            self.calculate_pagerank()
            self.save_pagerank_scores()
            logger.info("PageRank calculation and saving complete.")
        else:
            logger.warning("No URLs crawled. Skipping PageRank calculation.")
            
        self.db_conn.close()

        logger.info(f"Crawling complete. Total URLs crawled: {len(self.crawled_urls)}")

if __name__ == "__main__":
    starting_urls = ["https://fr.wikipedia.org/", "https://en.wikipedia.org/"]
    crawler = Crawler(starting_urls, max_urls=1000000, max_depth=5)
    try:
        asyncio.run(crawler.run())
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {str(e)}")