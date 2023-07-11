import sys
from wikipedia_parser import HEADER
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON


endpoint_url = "https://query.wikidata.org/sparql"

query="""SELECT ?person ?personLabel ?date_of_birth ?date_of_death ?image ?wikipedia_url WHERE {
  ?person wdt:P31 wd:Q5;
          wdt:P569 ?date_of_birth;
          wdt:P570 ?date_of_death;
          wdt:P18 ?image;
  FILTER(YEAR(?date_of_death) > 2015)
  ?article schema:about ?person;
           schema:isPartOf <https://en.wikipedia.org/>.  
  BIND(IRI(STR(?article)) AS ?wikipedia_url)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
  }
LIMIT 15000"""

def get_wikidata():
    """ returns DF with columns: date of birth, date of death, image link, wikipedia link """
    results = _get_results(endpoint_url, query)
    return _dict_to_dataframe(results['results']['bindings'])


def _get_results(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url, agent=HEADER['User-Agent'])
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def _dict_to_dataframe(dict_list):
    processed_dicts = []
    for d in dict_list:
        processed_dict = {}
        processed_dict['person'] = d['person']['value']
        processed_dict['date_of_death'] = d['date_of_death']['value']
        processed_dict['image'] = d['image']['value']
        processed_dict['date_of_birth'] = d['date_of_birth']['value']
        processed_dict['person_name'] = d['personLabel']['value']
        processed_dict['wikipedia_url'] = d['wikipedia_url']['value']
        processed_dicts.append(processed_dict)
    return pd.DataFrame(processed_dicts)


if __name__ == '__main__': 
    df = get_wikidata()

