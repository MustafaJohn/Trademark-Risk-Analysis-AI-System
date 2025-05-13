'''

This script exports trademark metadata and images from a PostgreSQL database with a limit of 1000 - this can be expanded in live or built upon further.
It saves the metadata to a CSV file and decodes the base64-encoded images into a local folder for indexing or model input.

'''



import os
import pandas as pd
import base64
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# --- Load environment variables ---
load_dotenv()

# --- DB Params from dotenv ---
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD")) 

# ---Build database URL ---
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Output folder ---
output_folder = "decoded_images"
os.makedirs(output_folder, exist_ok=True)

# --- query ---
query = """
SELECT 
    m.trademark_id,
    m.verbal_element_text,
    i.image_uri,
    i.file AS base64_image,
    t.application_number,
    t.classification_version
FROM images i
JOIN marks m ON i.mark_id = m.id
JOIN trademarks t ON m.trademark_id = t.id
LIMIT 1000;
"""

try:
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql_query(query, engine)

    # --- Save CSV ---
    df.drop(columns=["base64_image"]).to_csv("trademark_images_export.csv", index=False)
    print("✅ CSV saved as 'trademark_images_export.csv'")

    #  ---Decode and save images ---
    for idx, row in df.iterrows():
        base64_data = row['base64_image']
        if base64_data:
            try:
                image_data = base64.b64decode(base64_data)
                filename = f"{row['trademark_id']}.png"
                filepath = os.path.join(output_folder, filename)
                with open(filepath, "wb") as f:
                    f.write(image_data)
            except Exception as decode_err:
                print(f"⚠️ Failed to decode image at row {idx}: {decode_err}")

    print(f"✅ {len(df)} images processed and saved to '{output_folder}'")

except Exception as e:
    print("❌ Error:")
    print(e)
