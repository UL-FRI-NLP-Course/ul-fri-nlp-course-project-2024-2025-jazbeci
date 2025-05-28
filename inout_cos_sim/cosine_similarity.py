# This script calculates the cosine similarity between lemmatized input and output texts
# that we have created in lemmatize_classla.py
# the pairs are stored in a CSV file

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import sys


def pairwise_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        return df
    except Exception as e:
        # exit the program if the file cannot be read
        sys.exit(f"Failed to read CSV file {file_path}: {e}")

def process_lem_pair(lem_in, lem_out, lem_in_ner, lem_out_ner):
    cos_sim = pairwise_similarity(lem_in, lem_out)
    if lem_in_ner.strip() and lem_out_ner.strip():
        try:
            cos_sim_ner = pairwise_similarity(lem_in_ner, lem_out_ner)
        except ValueError:
            cos_sim_ner = float('nan')
    else:
        cos_sim_ner = float('nan')
    return {
        'LemCosineSimilarity': cos_sim,
        'LemCosineSimilarityNER': cos_sim_ner
    }

def process_pair(input_text, output_text):
    cos_sim = pairwise_similarity(input_text, output_text)
    return {
        'CosineSimilarity': cos_sim
    }

def process_and_save(lemmatized_file_name, file_name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    # calculate cosine similarity for each pair of lemmatized input and output texts
    # append the results to the dataframe
    try:
        logging.info(f"Processing {lemmatized_file_name}")
        df = read_csv_file(lemmatized_file_name)
        df = df.reset_index(drop=True)  # Reset index to ensure proper indexing after reading CSV
        # df has columns 'Datum', 'FilePath', 'InputText', 'OutputText', 'Input', 'Output', 'lem_in', 'lem_out', 'lem_in_ner', 'lem_out_ner'
        df['CosineSimilarity'] = 0.0
        # df['CosineSimilarityNER'] = 0.0
        # pairs_list = df[['lem_in', 'lem_out', 'lem_in_ner', 'lem_out_ner']].values.tolist()

        # df has columns 'Datum', 'FilePath', 'InputText', 'OutputText', 'Input', 'Output', 'IncludedInputKeys'
        pairs_list = df[['InputText', 'OutputText']].values.tolist()
        with ProcessPoolExecutor(max_workers=6) as executor:
            # futures = {executor.submit(process_pair, lem_in, lem_out, lem_in_ner, lem_out_ner): idx for idx, (lem_in, lem_out, lem_in_ner, lem_out_ner) in enumerate(pairs_list)}
            futures = {executor.submit(process_pair, input_text, output_text): idx for idx, (input_text, output_text) in enumerate(pairs_list)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    df.at[idx, 'CosineSimilarity'] = result['CosineSimilarity']
                    # df.at[idx, 'CosineSimilarityNER'] = result['CosineSimilarityNER']
                except Exception as e:
                    logging.error(f"Error processing pair at index {idx}: {e}")
        df.to_csv(file_name, index=False)
        logging.info(f"Cosine similarity results saved to {file_name}")
    except Exception as e:
        logging.error(f"Error processing file {lemmatized_file_name}: {e}")

if len(sys.argv) > 1:
    year = sys.argv[1]
else:
    year = '2024'

# input_file = f'./lemmatized_pairs_{year}.csv'
input_file = f'./pairs_{year}.csv'
output_file = f'./cos_sim_{year}.csv'

process_and_save(input_file, output_file)