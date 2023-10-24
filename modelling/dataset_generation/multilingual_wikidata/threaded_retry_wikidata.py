import sys
import os
import time
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import concurrent.futures

from retry_wikidata import *

endpoint_url = "https://query.wikidata.org/sparql"
OUTPUT_DIR = "."
TARGET_N = 100

query_template = """
SELECT DISTINCT ?person ?personLabel ?date_of_birth ?date_of_death ?image WHERE {{
  ?person wdt:P31 wd:Q5;
          wdt:P569 ?date_of_birth;
          wdt:P570 ?date_of_death;
          wdt:P18 ?image;
  FILTER(YEAR(?date_of_death) > 2010)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}
LIMIT 100
OFFSET {}
"""

def get_wikidata_threaded(start_offset):
    all_data = []
    offset = start_offset

    while True:
        query = query_template.format(offset)
        results = get_results_with_retry(endpoint_url, query)
        if not results["results"]["bindings"]:
            break  # Exit loop if no more results

        df = _dict_to_dataframe(results["results"]["bindings"])
        filename = f"CSVs/wikidata_offset_{offset}.csv"
        df.to_csv(filename, index=False)
        print(f"Thread-{start_offset} to {start_offset+999}: Got offset {offset} as CSV")

        all_data.append(df)
        offset += TARGET_N
        if offset >= start_offset + 1000:
            break

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def get_results_with_retry(endpoint_url, query):
    max_retries = 100
    retries = 0
    wait_times = [1, 3, 4]

    while retries < max_retries:
        try:
            return _get_results(endpoint_url, query)
        except Exception as e:
            retries += 1
            wait_time = wait_times[min(retries - 1, len(wait_times) - 1)]
            time.sleep(wait_time)
    raise Exception("Failed to fetch results after multiple retries.")


def _dict_to_dataframe(dict_list):
    processed_dicts = []
    for d in dict_list:
        processed_dict = {}
        processed_dict["person"] = d["person"]["value"]
        processed_dict["date_of_death"] = d["date_of_death"]["value"]
        processed_dict["image"] = d["image"]["value"]
        processed_dict["date_of_birth"] = d.get("date_of_birth", {}).get(
            "value", None
        )  # Handle potential missing date of birth
        processed_dict["person_name"] = d["personLabel"]["value"]
        # Handle potential missing Wikipedia URL
        processed_dict["wikipedia_url"] = d.get("wikipedia_url", {}).get("value", None)
        processed_dicts.append(processed_dict)
    return pd.DataFrame(processed_dicts)


def _get_results(endpoint_url, query):
    # Use a default user agent if not provided (Wikidata sometimes blocks requests without a user agent)
    user_agent = (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"
    )
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

if __name__ == '__main__':
    initial_offset = 47600
    batch_size = 1000
    total_threads = 3
    thread_offset_increment = batch_size * total_threads

    while True:
        all_dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(get_wikidata_threaded, initial_offset + i*batch_size) for i in range(total_threads)]
            for future in concurrent.futures.as_completed(futures):
                all_dfs.append(future.result())

        # If any of the threads return an empty dataframe, we'll assume there's no more data and break
        if any([df.empty for df in all_dfs]):
            break

        initial_offset += thread_offset_increment

