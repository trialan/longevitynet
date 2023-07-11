import requests
import re
import os
from bs4 import BeautifulSoup


FOLDER_PATH = os.getcwd()
HEADER = {'User-Agent': 'scraping a few hundred images for a personal project, contact: thomasrialan@gmail.com'}


def parse_wiki_page(url):
    person_name = url.split('/')[-1]
    rows = _get_info_box_rows(url)
    year_of_birth, year_of_death = _get_year_of_birth_and_death(rows)
    year_of_img = download_wiki_image(url, year_of_birth, year_of_death, person_name)
    return year_of_birth, year_of_death, year_of_img


def download_wiki_image(url, year_of_birth, year_of_death, person_name):
    response = requests.get(url, headers=HEADER)
    assert response.status_code == 200
    data = response.text
    soup = BeautifulSoup(data, 'html.parser')
    table = soup.find('table', {'class': 'infobox'})
    assert table, "Could not find the infobox."
    img = table.find('img')
    assert img, "Could not find the image."
    year_of_img = _get_year_of_image(img)
    img_url = 'https:' + img.get('src')
    response = requests.get(img_url, stream=True, headers=HEADER)
    response.raise_for_status()
    path = f'{FOLDER_PATH}/datasets/dataset_v3/{person_name}_birth:{year_of_birth}_death:{year_of_death}_data:{year_of_img}.jpg'
    assert year_of_img <= year_of_death
    assert year_of_img >= year_of_birth
    with open(path, 'wb') as out_file:
        #out_file.write(response.content)
        pass
    return year_of_img


def _get_year_of_image(img):
    candidate_captions = _get_candidate_captions(img)
    year_of_img = None
    for caption in candidate_captions:
        if _find_year(caption) is not None:
            year_of_img = _find_year(caption)
        elif len(_find_dates(caption)) > 0:
            year_of_img = int(_find_dates(img_caption)[0][:4])
    assert year_of_img is not None, "Could not find the year of the image."
    return year_of_img


def _get_candidate_captions(img):
    candidate_captions = [img.next_element.text, img.get('alt')]
    if img.find_next_sibling('div') is not None:
        candidate_captions = candidate_captions.append(img.find_next_sibling('div').text)
    return candidate_captions



def _get_year_of_birth_and_death(rows):
    for row in rows:
        header = row.find('th')
        if header:
            if 'born' in header.text.lower():
                dates = _find_dates(row.text)
                year_of_birth = int(dates[0].split('-')[0])
            if 'died' in header.text.lower():
                dates = _find_dates(row.text)
                year_of_death = int(dates[0].split('-')[0])
    return year_of_birth, year_of_death


def _get_info_box_rows(url):
    """ Pages of people have a box of info on the right side of the page. """
    response = requests.get(url, headers=HEADER)
    soup = BeautifulSoup(response.text, 'html.parser')
    infobox = soup.find('table', attrs={'class': 'infobox'})

    if infobox is None:
        print("Could not find the infobox.")
        return None

    rows = infobox.find_all('tr')
    return rows


def _find_dates(input_string):
    """ Find dates formatted as YYYY-MM-DD in a string. """
    pattern = r'\b(\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]))\b'
    return re.findall(pattern, input_string)


def _find_year(input_string):
    match = re.search(r'\b(19[0-9]{2}|20[0-9]{2})\b', input_string)
    if match is None:
        return None
    return int(match.group())
