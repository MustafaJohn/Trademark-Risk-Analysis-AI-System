""" 

This script provides utility functions to fetch and decode trademark images from the PostgreSQL database using trademark IDs.

"""

import psycopg2
import base64
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

# ---  DB PArams from dotenv ---
conn_params = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

def fetch_image_by_trademark_id(trademark_id):
    try:
        conn = psycopg2.connect(**conn_params)
        with conn.cursor() as cur:
            #--- Get mark_id from marks table ---
            cur.execute("SELECT id FROM marks WHERE trademark_id = %s LIMIT 1;", (trademark_id,))
            mark_row = cur.fetchone()
            if not mark_row:
                return None
            mark_id = mark_row[0]

            # --- get base64 encodedimage from images table ---
            cur.execute("SELECT file FROM images WHERE mark_id = %s LIMIT 1;", (mark_id,))
            img_row = cur.fetchone()
        conn.close()
            # --- Decode base64 image
        if img_row and img_row[0]:
            img_bytes = base64.b64decode(img_row[0])
            return Image.open(BytesIO(img_bytes)).convert("RGB") 
    except Exception as e:
        print(f"[DB ERr] Could not fetch image for trademark_id {trademark_id}: {e}")
        return None
