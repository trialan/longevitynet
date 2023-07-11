import requests
from bs4 import BeautifulSoup
import tqdm
import time


def get_people_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    base_url = 'https://en.wikipedia.org'

    people_links = []
    all_links = soup.find_all('a', href=True)

    for link in tqdm.tqdm(all_links):
        if '/wiki/' in link['href'] and ':' not in link['href']:
            time.sleep(0.3)
            possible_person_link = base_url + link['href']
            possible_person_page = requests.get(possible_person_link)
            soup_person = BeautifulSoup(possible_person_page.text, 'html.parser')
            bday = soup_person.find('span', {'class' : 'bday'})
            if bday:
                people_links.append(possible_person_link)
    
    return people_links


if __name__ == '__main__': 
    url = input("Enter a wikipedia url: ")
    links = get_people_links(url)
    with open("links.txt", "w") as f:
        for link in links:
            f.write(link + "\n")

