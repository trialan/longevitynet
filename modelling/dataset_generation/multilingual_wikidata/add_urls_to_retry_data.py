import os
import time
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

# Directory paths
input_directory = "CSVs"
output_directory = "CSVs_URLs"

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# SPARQL endpoint setup
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

def get_wikipedia_urls(batch_ids):
    """Query Wikidata SPARQL endpoint to get Wikipedia URLs for a batch of entity IDs."""
    values_clause = " ".join(f"wd:{entity_id}" for entity_id in batch_ids)
    query = f'''
    SELECT ?person ?wikipediaURL WHERE {{
      VALUES ?person {{ {values_clause} }}
      ?wikipediaURL schema:about ?person.
    }}
    '''
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    # Extract URLs for each ID
    urls = {}
    for result in results['results']['bindings']:
        person = result['person']['value'].split('/')[-1]
        url = result['wikipediaURL']['value']
        urls[person] = url
    return urls

files = sorted(os.listdir(input_directory), key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Process each CSV file in the input directory
for i, filename in tqdm(enumerate(files[84:])):
    if filename.endswith(".csv"):
        # Read CSV
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path)
        # Extract entity IDs from 'person' column
        entity_ids = df['person'].str.split('/').str[-1].tolist()
        # Fetch Wikipedia URLs in batches
        batch_size = 100
        all_urls = {}
        for start in range(0, len(entity_ids), batch_size):
            batch = entity_ids[start:start+batch_size]
            all_urls.update(get_wikipedia_urls(batch))
            time.sleep(2)  # Add a small delay between batches to prevent overloading the server
        
        # Map fetched URLs to dataframe
        df['wikipediaURL'] = df['person'].str.split('/').str[-1].map(all_urls)
        
        # Save results to new CSV in the output directory
        output_file_path = os.path.join(output_directory, filename)
        print(f"Got {filename} URLs")
        df[['person', 'wikipediaURL']].to_csv(output_file_path, index=False)

print("Processing complete!")
