import logging
from pathlib import Path
import os


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s: ')


lis_of_files = [
    "__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb"
]


for filepath in lis_of_files:
    filepath  =  Path(filepath)
    filedir, filename  = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating Directory named {filedir} for {filename}")

    if(not os.path.exists(filepath)) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating Empty File {filename}")

    else:

        logging.info(f"{filename} already exists")