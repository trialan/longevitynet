import pytest
from dataset_generation.wikipedia_parser import parse_wiki_page, _find_dates, _find_year


@pytest.mark.skip()
def test_parsing_a_page():
    url = "https://en.wikipedia.org/wiki/Kenneth_Kaunda"
    birthyear, deathyear, img_year = parse_wiki_page(url)
    assert_real_and_parsed_years_match_up(birthyear=1924,
                                          deathyear=2021,
                                          img_year=1983,
                                          parsed_birthyear=birthyear,
                                          parsed_deathyear=deathyear,
                                          parsed_img_year=img_year)

    url = "https://en.wikipedia.org/wiki/Gord_Brown"
    birthyear, deathyear, img_year = parse_wiki_page(url)
    assert_real_and_parsed_years_match_up(birthyear=1960,
                                          deathyear=2018,
                                          img_year=2018,
                                          parsed_birthyear=birthyear,
                                          parsed_deathyear=deathyear,
                                          parsed_img_year=img_year)

    url = "https://en.wikipedia.org/wiki/David_Bowie"
    birthyear, deathyear, img_year = parse_wiki_page(url)
    assert_real_and_parsed_years_match_up(birthyear=1947,
                                          deathyear=2016, 
                                          img_year=2002,
                                          parsed_birthyear=birthyear, 
                                          parsed_deathyear=deathyear,
                                          parsed_img_year=img_year)

    url = "https://en.wikipedia.org/wiki/Elvis_Presley"
    birthyear, deathyear, img_year = parse_wiki_page(url)
    assert_real_and_parsed_years_match_up(birthyear=1935, 
                                          deathyear=1977,
                                          img_year=1957,
                                          parsed_birthyear=birthyear, 
                                          parsed_deathyear=deathyear,
                                          parsed_img_year=img_year)

    url = "https://en.wikipedia.org/wiki/Chester_Bennington"
    birthyear, deathyear, img_year = parse_wiki_page(url)
    assert_real_and_parsed_years_match_up(birthyear=1976,
                                          deathyear=2017,
                                          img_year=2014,
                                          parsed_birthyear=birthyear,
                                          parsed_deathyear=deathyear,
                                          parsed_img_year=img_year)



def assert_real_and_parsed_years_match_up(birthyear, deathyear, img_year,
                                          parsed_birthyear, parsed_deathyear, parsed_img_year):
    assert birthyear == parsed_birthyear
    assert deathyear == parsed_deathyear
    assert img_year == parsed_img_year


def test_finding_dates():
    example = "asdkfljksdf 1960-04-04 askjf"
    assert _find_dates(example) == ["1960-04-04"]

    example = "asdsssilaskdlfljksdf 1988-04-09 askjf"
    assert _find_dates(example) == ["1988-04-09"]

    example = "asdsssilaskdlfljksdf"
    assert _find_dates(example) == []

def test_finding_year():
    example = "Gordon Brown lalalal 2018"
    assert _find_year(example) == 2018
