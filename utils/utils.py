import pandas as pd
import logging
from pathlib import Path
import os
import sys


def load_cctld_list(filename="country-codes-tlds.csv"):
    path = os.path.join(os.path.dirname(__file__), filename)
    cctld = pd.read_csv(path)[' tld'].apply(lambda x: x[2:]).tolist()
    return cctld


CCTLD = load_cctld_list()


def set_logger(log_path):
    """ Set Logger
    Args:
        log_path [str]
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="a")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("-----------------------------------------------")
    logger.info(f"Logger Configuration Done.")
    return logger


def get_labels_counts_map(df):
    labels_counts_map = {}
    for idx, row in df.iterrows():
        for label in row['target']:
            if label in labels_counts_map:
                labels_counts_map[label] += 1
            else:
                labels_counts_map[label] = 1
    return labels_counts_map
