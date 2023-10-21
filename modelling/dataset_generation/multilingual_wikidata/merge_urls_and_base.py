import pandas as pd
import os

url_data = os.listdir("CSVs_URLs/")
other_data = os.listdir("CSVs/")

def get_wikidata_df():
    url_df = pd.concat([pd.read_csv(f"CSVs_URLs/{f}") for f in url_data])
    url_df.rename(columns={"wikipediaURL": "wikipedia_url"}, inplace=True)
    url_df = url_df.drop_duplicates(subset=['wikipedia_url'])
    other_df = pd.concat([pd.read_csv(f"CSVs/{f}") for f in other_data])
    other_df = other_df.drop(columns=["wikipedia_url"])
    other_df = other_df.drop_duplicates(subset=['person'])
    df = other_df.merge(url_df, left_on='person', right_on='person', how='inner')
    return df


