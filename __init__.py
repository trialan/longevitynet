import os

def get_repo_dir():
    if "paperspace" in os.getcwd():
        REPO_DIR = "/home/paperspace/life_expectancy"
    else:
        REPO_DIR = "/Users/thomasrialan/Documents/code/longevity_project/life_expectancy"
    return REPO_DIR
