import pandas as pd
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

