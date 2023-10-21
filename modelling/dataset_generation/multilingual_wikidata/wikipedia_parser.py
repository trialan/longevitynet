import requests
import re
import os
from bs4 import BeautifulSoup

HEADER = {'User-Agent': 'scraping a few thousand images for a personal project, contact: thomasrialan@gmail.com'}
FOLDER = "/Users/thomasrialan/Documents/code/longevity_project/life_expectancy/modelling/dataset_generation/dataset_multilingual_v1"

def download_wiki_image(url, year_of_birth, year_of_death, person_name):
    response = requests.get(url, headers=HEADER)
    assert response.status_code == 200
    data = response.text
    soup = BeautifulSoup(data, 'html.parser')
    results = refined_extract_images_and_captions(soup)
    saved_images = []

    for img_url, caption, year_of_img in results:
        if not year_of_img:
            print("No year of image")
            print(img_url)
            print(url)
            print(caption)
            print("---------------")
        if not (int(year_of_birth) <= int(year_of_img) <= int(year_of_death)):
            print("Years not ordered")
            print(img_url)
            print(url)
            print(caption)

        assert year_of_img
        assert int(year_of_birth) <= int(year_of_img) <= int(year_of_death)

        image_response = requests.get(img_url, headers=HEADER)
        assert image_response.status_code == 200
        file_path = f"{FOLDER}/{person_name}_birth:{year_of_birth}_death:{year_of_death}_data:{year_of_img}.jpg"
        print(file_path)
        with open(file_path, 'wb') as img_file:
            img_file.write(image_response.content)


def refined_extract_images_and_captions(soup):
    images = soup.find_all('img')
    results = []

    for img in images:
        img_url = 'https:' + img.get('src').split(" ")[0]
        if any(term in img_url for term in ["logo", "svg", "static", "icons", "copyright"]):
            continue

        caption_element = img.find_next(['figcaption', 'div'])
        if caption_element:
            caption = caption_element.text.strip()
            if 10 < len(caption) < 500 and not caption.startswith('Obtenido de'):
                year_of_img = _find_year(caption)
                if not year_of_img and len(_find_dates(caption)) > 0:
                    year_of_img = int(_find_dates(caption)[0][:4])
                results.append((img_url, caption, year_of_img))
    out = [r for r in results if r[2] is not None]
    return out


def _find_year(input_string):
    """Find a year in the format YYYY in a string."""
    match = re.search(r'\b(19[0-9]{2}|20[0-9]{2})\b', input_string)
    if match is None:
        return None
    return int(match.group())


def _find_dates(input_string):
    """Find dates formatted as YYYY-MM-DD in a string."""
    pattern = r'\b(\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]))\b'
    return re.findall(pattern, input_string)


if __name__ == '__main__':
    from tqdm import tqdm
    from multilingual_wikidata.merge_urls_and_base import get_wikidata_df

    df = get_wikidata_df()

    errors = []
    DLs = 0

    rows = list(df.iterrows())

    for ix, row in enumerate(rows):
        row = row[1]
        print(f"***** INDEX {ix} *****")

        if type(row['wikipedia_url']) != str:
            continue

        if "wikimedia" in row['wikipedia_url']:
            continue

        try:
            birth_year = row['date_of_birth'][:4]
            death_year = row['date_of_death'][:4]
            download_wiki_image(row['wikipedia_url'],
                                birth_year,
                                death_year,
                                row['person_name'])
            DLs += 1
        except:
            errors.append(ix)
            print(row['wikipedia_url'])
            print(f"Errors: {len(errors)}")
            print(f"DLs: {DLs}")


