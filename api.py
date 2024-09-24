from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import math
import numpy as np
import logging
import base64

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    conn = sqlite3.connect('crawler_data.db')
    conn.row_factory = sqlite3.Row
    return conn

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.get("/search")
async def search(query: str, limit: int = 10):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    
    logger.info(f"Received search query: {query}")
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        query_terms = query.lower().split()
        placeholders = ', '.join(['?' for _ in query_terms])
        
        logger.info("Executing search query")
        cursor.execute(f"""
            SELECT DISTINCT inverted_index.url, pages.title, pages.category, 
                   SUM(inverted_index.tf_idf) as total_score, 
                   inverted_index.semantic_vector
            FROM inverted_index
            JOIN pages ON inverted_index.url = pages.url
            WHERE inverted_index.word IN ({placeholders})
            GROUP BY inverted_index.url
            ORDER BY total_score DESC
            LIMIT ?
        """, query_terms + [limit])

        results = []
        for row in cursor.fetchall():
            logger.info(f"Processing result: {row['url']}")
            semantic_vector = row['semantic_vector']
            if semantic_vector:
                try:
                    semantic_vector = np.frombuffer(semantic_vector, dtype=np.float32)
                    semantic_similarity = cosine_similarity(semantic_vector, semantic_vector).item()
                except Exception as e:
                    logger.error(f"Error processing semantic vector for {row['url']}: {str(e)}")
                    semantic_similarity = 0
            else:
                semantic_similarity = 0
            
            results.append({
                "url": row['url'],
                "title": row['title'],
                "category": row['category'],
                "score": float(row['total_score']),
                "semantic_similarity": semantic_similarity
            })

        # Fetch PageRank scores
        urls = [result['url'] for result in results]
        url_placeholders = ', '.join(['?' for _ in urls])
        logger.info("Fetching PageRank scores")
        cursor.execute(f"""
            SELECT url, score
            FROM pagerank
            WHERE url IN ({url_placeholders})
        """, urls)
        
        pagerank_scores = {row['url']: row['score'] for row in cursor.fetchall()}

        # Combine scores
        for result in results:
            result['pagerank'] = float(pagerank_scores.get(result['url'], 0))
            result['combined_score'] = (result['score'] * 0.4 + 
                                        result['semantic_similarity'] * 0.3 + 
                                        result['pagerank'] * 0.3)

        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)

        logger.info(f"Returning {len(results)} results")
        return {"results": results}
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        conn.close()

@app.get("/categories")
async def get_categories():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM pages")
        categories = [row['category'] for row in cursor.fetchall()]
        return {"categories": categories}
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)