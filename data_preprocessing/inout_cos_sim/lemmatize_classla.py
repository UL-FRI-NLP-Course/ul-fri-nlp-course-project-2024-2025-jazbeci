
# ## WARNING! for this file you need classla. you might get conflicts with tensorflow so i suggest you use a separate python environment for this one
# 
# in this notebook we are tokenizin input text (text from promet.si up to one hour before the reading of traffic news) and the output (the traffic news)
# 
# so if we have traffic news from 1.1.2024 11:30, we will have as input texts all the rows of the excel file (probably web scraped) that are from 1.1.2024 10:30 to 11:30

import classla
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import logging
import sys

# Global variable for the pipeline
nlp = None

def init_pipeline():
    global nlp
    nlp = classla.Pipeline('sl', processors='tokenize,ner,pos,lemma')


def lemma_string(doc):
  list_of_lemmas = [word.lemma for t in doc.iter_tokens() for word in t.words]
  odstrani_locila = [x for x in list_of_lemmas if x not in [',','.', ':', '-', ';']]
  list_to_string = ' '.join(odstrani_locila)
  return list_to_string

def lemma_string_ner(doc):
    list_of_lemmas = []
    for sentence in doc.sentences:
        for t, w in zip(sentence.tokens, sentence.words):
            if(t.ner != 'O'):
                list_of_lemmas.append(w.lemma)
    return ' '.join(list_of_lemmas)

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        return df
    except Exception as e:
        # exit the program if the file cannot be read
        sys.exit(f"Failed to read CSV file {file_path}: {e}")

# from the pairs we made with pairs.py we will lemmatize the input and output texts so that we can later compare them with cosine similarity (tf-idf)
def process_pair(input_text, output_text):
    global nlp
    # input_text is a string with the input text (promet.si) and output_text is a string with the output text (student report)
    tokenized_input = nlp(input_text)
    tokenized_output = nlp(output_text)
    lem_in = lemma_string(tokenized_input)
    lem_out = lemma_string(tokenized_output)
    lem_in_ner = lemma_string_ner(tokenized_input)
    lem_out_ner = lemma_string_ner(tokenized_output)
    return {
        'lem_in': lem_in,
        'lem_out': lem_out,
        'lem_in_ner': lem_in_ner,
        'lem_out_ner': lem_out_ner
    }

def lemmatize_pairs(pairs_df, n_jobs=6):
    # pairs_df have columns 'Datum', 'FilePath', 'InputText', 'OutputText', 'Input', 'Output'
    # divide work of lemmatization into n_jobs processes (we divide rows of the dataframe into n_jobs chunks)
    # keep the indexes of the original dataframe to be able to merge results later into the pairs_df
    pairs_df['lem_in'] = ''
    pairs_df['lem_out'] = ''
    pairs_df['lem_in_ner'] = ''
    pairs_df['lem_out_ner'] = ''
    pairs_list = pairs_df[['InputText', 'OutputText']].values.tolist()
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=init_pipeline) as executor:
        futures = {executor.submit(process_pair, input_text, output_text): idx for idx, (input_text, output_text) in enumerate(pairs_list)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                pairs_df.at[idx, 'lem_in'] = result['lem_in']
                pairs_df.at[idx, 'lem_out'] = result['lem_out']
                pairs_df.at[idx, 'lem_in_ner'] = result['lem_in_ner']
                pairs_df.at[idx, 'lem_out_ner'] = result['lem_out_ner']
            except Exception as e:
                # log also the Datum and FilePath for better debugging
                logging.error(f"Error processing pair at datum {pairs_df.at[idx, 'Datum']} and FilePath {pairs_df.at[idx, 'FilePath']}: {e}")
    
    return pairs_df


def process_and_save(pairs_file_name, file_name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    try:
        logging.info(f"Processing {pairs_file_name}")
        pairs = read_csv_file(pairs_file_name)
        df = lemmatize_pairs(pairs, n_jobs=6)
        df.to_csv(file_name, index=False)
        logging.info(f"Lemmatized data saved to {file_name}")
    except Exception as e:
        logging.error(f"Error lemmatizing {pairs_file_name}: {e}")

if len(sys.argv) > 1:
    year = sys.argv[1]
else:
    year = '2024'

input_file = f'./pairs_{year}.csv'
output_file = f'./lemmatized_pairs{year}.csv'

process_and_save(input_file, output_file)