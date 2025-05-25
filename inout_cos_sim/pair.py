
# make pairs of input and output text for later cosine similarity calculation
# and save them to a CSV file


import json

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import logging
import sys


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def make_pairs_pd(json_data, n_jobs=6):
    # Collect all pairs for the whole file
    all_pairs = []
    for item in json_data:
        if 'input' in item and 'output' in item:
            output_path = item['output']['FilePath']
            output_text = item['output']['Content'].split('\n', 1)[1]
            for input in item['input']:
                if 'Datum' in input:
                    datum = input['Datum']
                    input_text = ' '.join([input[key] for key in input.keys() if key != 'Datum'])
                    all_pairs.append((datum, output_path, input_text, output_text, input, item['output']))
    # Create a DataFrame from the collected pairs
    df = pd.DataFrame(all_pairs, columns=['Datum', 'FilePath', 'InputText', 'OutputText', 'Input', 'Output'])
    return df

def process_and_save(json_data_file_name, file_name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    try:
        logging.info(f"Processing {json_data_file_name}")
        json_data = read_json_file(json_data_file_name)
        df = make_pairs_pd(json_data, n_jobs=6)
        df.to_csv(file_name, index=False)
        logging.info(f"Pairs saved to {file_name}")
    except Exception as e:
        logging.error(f"Error making pairs {json_data_file_name}: {e}")

if len(sys.argv) > 1:
    year = sys.argv[1]
else:
    year = '2024'

input_file = f'../Processed/input_output_all_data_{year}_reduced.json'
output_file = f'./pairs_{year}.csv'

process_and_save(input_file, output_file)