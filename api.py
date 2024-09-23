from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import math

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

@app.get("/search")
async def search(query: str, limit: int = 10):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        query_terms = query.lower().split()
        placeholders = ', '.join(['?' for _ in query_terms])
        
        cursor.execute(f"""
            SELECT DISTINCT inverted_index.url, pages.title, SUM(inverted_index.tf_idf) as total_score
            FROM inverted_index
            JOIN pages ON inverted_index.url = pages.url
            WHERE inverted_index.word IN ({placeholders})
            GROUP BY inverted_index.url
            ORDER BY total_score DESC
            LIMIT ?
        """, query_terms + [limit])

        results = [{"url": row['url'], "title": row['title'], "score": row['total_score']} for row in cursor.fetchall()]

        return {"results": results}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)