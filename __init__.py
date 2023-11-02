import os

def get_repo_dir():
    if "paperspace" in os.getcwd():
        REPO_DIR = "/home/paperspace/longevitynet"
    elif "root" in os.getcwd():
        REPO_DIR = "/root/longevitynet"
    else:
        REPO_DIR = "/Users/thomasrialan/Documents/code/longevitynet"
    return REPO_DIR

REPO_DIR = get_repo_dir()
