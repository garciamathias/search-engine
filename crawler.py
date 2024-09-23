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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import ssl

# Configuration SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Téléchargement des ressources NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Crawler:
    def __init__(self, starting_urls, max_urls=10000, num_workers=5, delay=1):
        self.starting_urls = starting_urls
        self.max_urls = max_urls
        self.num_workers = num_workers
        self.delay = delay
        self.to_crawl = asyncio.Queue()
        self.mini_queue = deque(maxlen=1000)
        self.crawled_urls = set()
        self.domain_last_crawled = {}
        self.rp_cache = {}
        self.session = None
        self.db_conn = None
        self.stop_words = set(stopwords.words('english') + stopwords.words('french'))
        self.should_stop = asyncio.Event()

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
                    content TEXT
                )
            ''',
            'queue': '''
                CREATE TABLE IF NOT EXISTS queue (
                    url TEXT PRIMARY KEY
                )
            ''',
            'inverted_index': '''
                CREATE TABLE IF NOT EXISTS inverted_index (
                    word TEXT,
                    url TEXT,
                    tf_idf REAL,
                    PRIMARY KEY (word, url)
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

        self.db_conn.commit()
        cursor.close()

    async def fetch(self, url, session):
        try:
            async with session.get(url, timeout=10, ssl=False) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
        return None

    def parse(self, html, url):
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.title.string if soup.title else ""
            content = soup.get_text()
            links = [urljoin(url, link.get('href')) for link in soup.find_all('a', href=True)]
            return title, content, links
        except Exception as e:
            logger.error(f"Error parsing {url}: {str(e)}")
            return "", "", []

    async def crawl_page(self, url):
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

        html = await self.fetch(url, self.session)
        if html:
            title, content, links = self.parse(html, url)
            self.save_to_db(url, title, content)
            self.crawled_urls.add(url)
            self.domain_last_crawled[domain] = time.time()

            for link in links:
                if link not in self.crawled_urls and len(self.crawled_urls) < self.max_urls:
                    await self.add_to_queue(link)

            await self.index_page(url, title, content)

        logger.info(f"Crawled: {url}")

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

    def save_to_db(self, url, title, content):
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO pages (url, title, content) VALUES (?, ?, ?)",
                           (url, title, content))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")

    async def add_to_queue(self, url):
        self.mini_queue.append(url)
        if len(self.mini_queue) >= 100:
            await self.flush_queue_to_db()

    async def flush_queue_to_db(self):
        if not self.mini_queue:
            return
        try:
            cursor = self.db_conn.cursor()
            cursor.executemany("INSERT OR IGNORE INTO queue (url) VALUES (?)",
                               [(url,) for url in self.mini_queue])
            self.db_conn.commit()
            self.mini_queue.clear()
        except Exception as e:
            logger.error(f"Error flushing queue to database: {str(e)}")

    async def get_next_url(self):
        if not self.mini_queue:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT url FROM queue LIMIT 1000")
                urls = cursor.fetchall()
                if urls:
                    self.mini_queue.extend([url[0] for url in urls])
                    cursor.execute("DELETE FROM queue WHERE url IN (" + ",".join("?" * len(urls)) + ")",
                                   [url[0] for url in urls])
                    self.db_conn.commit()
            except Exception as e:
                logger.error(f"Error getting next URL from database: {str(e)}")

        return self.mini_queue.popleft() if self.mini_queue else None

    async def index_page(self, url, title, content):
        try:
            try:
                words = word_tokenize(f"{title} {content}")
            except LookupError:
                words = (f"{title} {content}").split()
            
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
                
                cursor.execute("INSERT OR REPLACE INTO inverted_index (word, url, tf_idf) VALUES (?, ?, ?)",
                            (word, url, tf_idf))
            
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error indexing page {url}: {str(e)}")

    async def worker(self):
        while not self.should_stop.is_set():
            try:
                url = await self.get_next_url()
                if url is None:
                    if self.to_crawl.empty():
                        break
                    url = await self.to_crawl.get()
                    process_queue_item = True
                else:
                    process_queue_item = False
                
                if url not in self.crawled_urls:
                    await self.crawl_page(url)
                
                if process_queue_item:
                    self.to_crawl.task_done()
            except Exception as e:
                logger.error(f"Error in worker: {str(e)}")
                if process_queue_item:
                    self.to_crawl.task_done()

    async def run(self):
        logger.info("Starting crawler...")
        await self.init_db()
        self.session = aiohttp.ClientSession()

        for url in self.starting_urls:
            await self.to_crawl.put(url)

        workers = [asyncio.create_task(self.worker()) for _ in range(self.num_workers)]

        await asyncio.gather(*workers)

        await self.flush_queue_to_db()
        await self.session.close()
        self.db_conn.close()

        logger.info(f"Crawling complete. Total URLs crawled: {len(self.crawled_urls)}")

if __name__ == "__main__":
    starting_urls = ["https://fr.wikipedia.org/", "https://fr.wikipedia.org/"]
    crawler = Crawler(starting_urls, max_urls=10000)
    try:
        asyncio.run(crawler.run())
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {str(e)}")