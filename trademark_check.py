import psycopg2
import pandas as pd
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from metaphone import doublemetaphone
import argparse
from rapidfuzz import process, fuzz

# ----------------------- Configuration Constants ----------------------- #
DB_HOST = db_host
DB_PORT = db_port
DB_NAME = your_db_name
DB_USER = your_db_user
DB_PASSWORD = password

JUNK_WORDS = ['limited', 'ltd', 'inc', 'plc', 'group', 'company', 'co', 'uk', 'llp']

# These values are decided after some testing and are not set in stone.
BERT_THRESHOLD = 0.5
RAPIDFUZZ_THRESHOLD = 60
DEFAULT_BATCH_SIZE = 100000
DEFAULT_MAX_BATCHES = 50

# ----------------------- Connect to Database ----------------------- #
try:
    print("[INFO] Connecting to the database...")
    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    print("[INFO] Connected to the database successfully.")
except Exception as e:
    print(f"[ERROR] Failed to connect to database: {e}")
    exit(1)

# ----------------------- Functions ----------------------- #

# Basic preprocesing done on the company names 
def clean_name(name):
    tokens = name.lower().replace('.', '').split()
    return ' '.join([t for t in tokens if t not in JUNK_WORDS])

# Function to calculate the soundslike score using doublemetaphone
def soundslike_score(name1, name2):
    mp1 = doublemetaphone(name1)
    mp2 = doublemetaphone(name2)
    
    # Calculate the soundslike score using RapidFuzz
    scores = [fuzz.ratio(code1, code2) for code1 in mp1 for code2 in mp2 if code1 and code2]
    
    # Returning the scores divided by 10 to scale it down to a 0-10 range
    return max(scores)/10 if scores else 0

# ----------------------- Fetch Company Data -----------------------
def fetch_data(batch_size=DEFAULT_BATCH_SIZE, last_id=0):
    
    # Query to fetch company data from the database
    query = f"""
            SELECT trademark_id, verbal_element_text
            FROM marks
            WHERE feature = 'Word'
            AND trademark_id > %s
            ORDER BY trademark_id ASC  
            LIMIT %s;
            """
    # Doing a rollback to ensure the connection is in a clean state before executing the query
    conn.rollback()
    
    # Executing the query and fetching the results
    with conn.cursor() as cur:
        cur.execute(query, (last_id, batch_size))
        return cur.fetchall()

# ----------------------- Matching & Scoring ----------------------- #

# Function to get candidates using RapidFuzz. RapidFuzz is used for fuzzy string matching.
def get_rapidfuzz_candidates(input_name, candidate_names, threshold=RAPIDFUZZ_THRESHOLD, limit=50):
    
    # Preprocess the input name to remove junk words and split into keywords
    input_clean = clean_name(input_name)
    input_keywords = set(input_clean.split())
    matches = []

    for original_name in candidate_names:
        if original_name: 
            cleaned = clean_name(original_name)
            keyword_match_score = sum(5 for kw in input_keywords if kw in cleaned.split())
            # Finding the score using RapidFuzz
            score = fuzz.token_sort_ratio(input_clean, cleaned)
            final_score = score + keyword_match_score

            # Only storing the matches that meet the threshold removing redundant ones
            if final_score >= threshold:
                matches.append((original_name, final_score))

    return sorted(matches, key=lambda x: x[1], reverse=True)[:limit]

# Defining the BERT model for reranking
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to rerank candidates using BERT embeddings
def rerank_with_bert(input_embedding, candidates, threshold=0.7, force_visual_score=90):
    
    if not candidates:
        return []

    candidate_names = [name for name, _ in candidates]

    if not candidate_names:
        return []

    # Extract names from candidate list
    candidate_names = [name for name, _ in candidates]

    # Encode candidate names using BERT
    candidate_embeddings = model.encode(candidate_names)

    # Compute cosine similarity between input and each candidate
    similarities = cosine_similarity(input_embedding, candidate_embeddings).flatten()

    reranked = []
    for (name, fuzzy_score), bert_score in zip(candidates, similarities):
        
        # If fuzzy score is very high but BERT score is low, override it
        if fuzzy_score >= force_visual_score and bert_score < threshold:
            adjusted_score = threshold
        else:
            adjusted_score = bert_score

        reranked.append((name, adjusted_score))

    # Sort results by adjusted score descending
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked

def get_final_info(trademark_id):
    # Query to fetch the final information based on the trademark ID
    query = f"""
            SELECT a.name, ad.country, at.active, nc.number
            FROM applicants a 
            JOIN applicant_trademarks at ON a.id = at.applicant_id
            JOIN nice_class_trademarks nt ON at.trademark_id = nt.trademark_id
            JOIN nice_classes nc ON nt.nice_class_id = nc.id
            JOIN addresses ad ON ad.addressable_id = a.id
            WHERE nt.trademark_id = {trademark_id}; 
            """
    # Doing a rollback to ensure the connection is in a clean state before executing the query
    conn.rollback()

    # Executing the query and fetching the results
    with conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchall()
    
    return result 

# ----------------------- Trademark Check ----------------------- #
def trademark_check(input_name, input_country, input_class, 
                        batch_size=DEFAULT_BATCH_SIZE,
                        max_batches=DEFAULT_MAX_BATCHES):
    
    offset = 0
    last_id = 0
    all_results = []
    total_processed = 0
    batches_processed = 0
    start_time = datetime.now()

    while True:
       
        # Fetching company data in batches to avoid memory issues 
        company_data = fetch_data(batch_size, last_id)

        if not company_data:
            print(f"[INFO] Processed {total_processed} records in total.")
            print("[INFO] No more data to process.", " "*20) 
            break

        print(f"[INFO] Processed {offset + len(company_data)} records so far...", end='\r')
        
        # This step helps to avoid duplicates in the company names by adding them to a set
        company_names = list(set([c[1] for c in company_data]))
        
        # Getting the candidates using RapidFuzz
        candidates = get_rapidfuzz_candidates(input_name, company_names)
        
        # Encoding the input name using BERT
        input_embedding = model.encode([input_name])

        # Reranking the candidates using BERT
        reranked = rerank_with_bert(input_embedding, candidates)

        # Filtering and iterating the reranked candidates based on the BERT threshold
        for company_name, score in reranked:

            if score < BERT_THRESHOLD:
                continue
            
            # Extracting the company information from the original data
            # Using next() to find the first match in the company data
            # This is done to avoid duplicates in the company names
            company_info = next((row for row in company_data if row[1] and company_name and row[1].strip().lower() == company_name.strip().lower()), None)

            if not company_info:
                continue
            
            trademark_id = company_info[0]

            # Getting the final information using the trademark ID
            final_info = get_final_info(trademark_id)            
            
            if final_info:
                # Extracting the company information and appending it to a dataframe 
                all_results.append({
                    'Input Name': input_name,
                    'Trademark': company_name,
                    #'Trademark ID': trademark_id, #For debugging purposes
                    'Trademark Owner': final_info[0][0],
                    'Active Status': final_info[0][2],
                    #'SIC Code': company_info[2],
                    'Country': final_info[0][1],
                    'Nice Classes': ",".join(str(x) for x in sorted(set(tup[3] for tup in final_info))),
                    'BERT Score': score * 10  # scaled
                })

        total_processed += len(company_data)
        offset += batch_size
        last_id = company_data[-1][0]
        batches_processed += 1

        # Terminating the loop if the maximum number of batches is reached
        if batches_processed >= max_batches:
            print(f"[INFO] Stopped after {max_batches} batches.")
            break

    if not all_results:
        print("[WARNING] No matches found.")
        return

    # ----------------------- Report generation ----------------------- #

    # Removing duplicates based on the company name
    df = pd.DataFrame(all_results).drop_duplicates(subset=['Trademark Owner'])

    # Calculating the scores for each company based on the rubric 
    df['Looks Like Score'] = df.apply(lambda row: fuzz.token_set_ratio(row['Trademark'].lower(), row['Input Name'].lower()) / 10, axis=1)
    df['Sounds Like Score'] = df.apply(lambda row: soundslike_score(row['Trademark'], row['Input Name']), axis=1)
    df['Classification Sector Score'] = df.apply(lambda row: 5 if set(input_class.split(',')) & set(row['Nice Classes'].split(',')) else 0, axis=1)
    df['Active Score'] = df.apply(lambda row: 0 if row['Active Status'] is True else -10, axis=1)
    df['Country Score'] = df.apply(lambda row: 10 if row['Country'] == input_country else 0, axis=1)

    # Calculating the final score by summing all the individual scores
    df['Final Score'] = df['BERT Score'] + df['Looks Like Score'] + df['Sounds Like Score'] + df['Active Score'] + df['Country Score'] + df['Classification Sector Score'] 
    df['Risk'] = df['Final Score'].apply(lambda x: 'Very High Risk' if x >= 40 else 'High Risk' if (x > 35 and x < 40) else 'Medium Risk' if (x < 35 and x > 25) else 'Low Risk')
    df = df.sort_values(by='Final Score', ascending=False, ignore_index=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{input_name}_report_{timestamp}.csv"
    try:
        df.to_csv(filename, index=False)
        duration = (datetime.now() - start_time).total_seconds()
        print(f"[SUCCESS] Report saved as {filename}. Processed {total_processed} companies in {duration:.2f} seconds.")
        print(f"[SUMMARY] Top 5 Matches for {input_name}:\n{df[['Trademark Owner', 'Final Score', 'Risk']].head()}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")

# ----------------------- Run  ----------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
                                    " _______________________________________________________________________________\n"
                                    "| AI-Driven Trademark Similarity Report Generator.\t\t\t\t|\n"
                                    "| Example:\t\t\t\t\t\t\t\t\t|\n"
                                    "|  python trademark_check.py -name \"Glass Resort Ltd\" -country GB -class 4\t|\n"
                                    "|_______________________________________________________________________________|\n\n"
                                    ), 
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    )

    parser.add_argument("-name", "--input_name", required=True, type=str,
                        help="Company name to check (use quotes if multi-word)")
    parser.add_argument("-country", "--input_country", required=True, type=str,
                        help="Country code of the input company (e.g. GB)")
    parser.add_argument("-class", "--input_class", required=True, type=str,
                        help="Nice class of the input company (e.g. 25, 35, etc.)")
    
    args = parser.parse_args()

    
    trademark_check(
        input_name=args.input_name,
        input_country=args.input_country,
        input_class=args.input_class
    )
