import tqdm
import time
import pandas as pd
from wikidata import get_wikidata
from wikipedia_parser import download_wiki_image, FOLDER_PATH
import json

def json_file_to_dataframe(file_path):
    # Open the JSON file and load it into a Python object
    with open(file_path, 'r') as f:
        json_obj = json.load(f)
    
    # Initialize an empty list to store individual rows
    data = []
    
    # Iterate over each entry in the "bindings" list
    for entry in json_obj['results']['bindings']:
        # Extract the data for each column
        person = entry['person']['value']
        date_of_death = entry['date_of_death']['value']
        image = entry['image']['value']
        date_of_birth = entry['date_of_birth']['value']
        person_name = entry['personLabel']['value']
        wikipedia_url = entry['wikipedia_url']['value']
        
        # Add this data as a new row in our list
        data.append([person, date_of_death, image, date_of_birth, person_name, wikipedia_url])
    
    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(data, columns=['person', 'date_of_death', 'image', 'date_of_birth', 'person_name', 'wikipedia_url'])
    
    return df


if __name__ == '__main__': 
    df = json_file_to_dataframe('fourteen_thousand_rows.json')
    errors = []

    for i in tqdm.tqdm(range(len(df))):
        try:
            url = df['wikipedia_url'][i]
            birth_year = int(df['date_of_birth'][i].split('-')[0])
            death_year = int(df['date_of_death'][i].split('-')[0])
            person_name = df['person_name'][i].replace(" ","_")
            download_wiki_image(url, birth_year, death_year, person_name)
        except:
            errors.append(url)
            print(len(errors))


