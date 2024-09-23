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

# Désactiver la vérification SSL pour éviter les erreurs de certificat
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Crawler:
    def __init__(self, starting_urls, max_urls=1000, num_workers=5, delay=1):
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
        self.stop_words = set(stopwords.words('english'))

    async def init_db(self):
        self.db_conn = sqlite3.connect('crawler_data.db')
        cursor = self.db_conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS pages
                          (url TEXT PRIMARY KEY, title TEXT, content TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS queue
                          (url TEXT PRIMARY KEY)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS inverted_index
                          (word TEXT, url TEXT, tf_idf REAL,
                           PRIMARY KEY (word, url))''')
        self.db_conn.commit()

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
                
                cursor.execute("INSERT OR REPLACE INTO inverted_index (word, url, tf_idf) VALUES (?, ?, ?)",
                               (word, url, tf_idf))
            
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error indexing page {url}: {str(e)}")

    async def worker(self):
        while True:
            try:
                url = await self.get_next_url()
                if url is None:
                    if self.to_crawl.empty():
                        break
                    url = await self.to_crawl.get()
                
                if url not in self.crawled_urls:
                    await self.crawl_page(url)
                
                if not self.to_crawl.empty():
                    self.to_crawl.task_done()
            except Exception as e:
                logger.error(f"Error in worker: {str(e)}")

    async def run(self):
        logger.info("Starting crawler...")
        await self.init_db()
        logger.info("Database initialized")
        self.session = aiohttp.ClientSession()
        logger.info("Session created")

        for url in self.starting_urls:
            await self.to_crawl.put(url)
            logger.info(f"Added starting URL to queue: {url}")

        workers = [asyncio.create_task(self.worker()) for _ in range(self.num_workers)]

        await self.to_crawl.join()
        await self.flush_queue_to_db()

        for worker in workers:
            worker.cancel()

        await self.session.close()
        self.db_conn.close()

        logger.info(f"Crawling complete. Total URLs crawled: {len(self.crawled_urls)}")

if __name__ == "__main__":
    starting_urls = ["https://lemonde.fr", "https://www.lefigaro.fr/"]
    crawler = Crawler(starting_urls)
    try:
        asyncio.run(crawler.run())
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {str(e)}")