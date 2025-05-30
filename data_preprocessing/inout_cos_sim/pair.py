
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

def process_chunk(chunk):
    pairs = []
    for idx, item in chunk:
        if 'input' in item and 'output' in item:
            output_path = item['output']['FilePath']
            output_text = item['output']['Content'].split('\n', 1)[1] # remove first line of student report
            for input in item['input']:
                if 'Datum' in input:
                    datum = input['Datum']
                    input_keys = list(input.keys())
                    not_included_keys = ['Datum', 'A2', 'B2', 'C2']
                    for key in not_included_keys:
                        if key in input_keys:
                            input_keys.remove(key)
                    if len(input_keys) == 0:
                        logging.warning(f"No input keys left after removing {not_included_keys} for item with Datum: {datum}")
                        continue  # Skip if no relevant input data remain
                    input_text = ' '.join([input[key] for key in input_keys])
                    pairs.append((idx, datum, output_path, input_text, output_text, input, item['output'], input_keys))
    return pairs

def chunkify(lst, n):
    # lst is now a list of (idx, item)
    return [lst[i::n] for i in range(n)]

def make_pairs_pd(json_data, n_jobs=6):
    # Add index to each item
    indexed_data = list(enumerate(json_data))
    all_pairs = []
    chunks = chunkify(indexed_data, n_jobs)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in futures:
            all_pairs.extend(future.result())
    df = pd.DataFrame(all_pairs, columns=['Index', 'Datum', 'FilePath', 'InputText', 'OutputText', 'Input', 'Output', 'IncludedInputKeys'])
    df = df.sort_values('Index').reset_index(drop=True)
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