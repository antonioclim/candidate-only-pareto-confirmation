
import json
import pandas as pd
from common.style import DATA_DIR

def read_csv(name):
    return pd.read_csv(DATA_DIR / name)

def read_json(name):
    with open(DATA_DIR / name, 'r', encoding='utf-8') as f:
        return json.load(f)
