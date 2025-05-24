
# ## WARNING! for this file you need classla. you might get conflicts with tensorflow so i suggest you use a separate python environment for this one
# 
# in this notebook we are tokenizin input text (text from promet.si up to one hour before the reading of traffic news) and the output (the traffic news)
# 
# so if we have traffic news from 1.1.2024 11:30, we will have as input texts all the rows of the excel file (probably web scraped) that are from 1.1.2024 10:30 to 11:30

import classla
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

def pairwise_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def lemma_string_ner(doc):
    list_of_lemmas = []
    for sentence in doc.sentences:
        for t, w in zip(sentence.tokens, sentence.words):
            if(t.ner != 'O'):
                list_of_lemmas.append(w.lemma)
    return ' '.join(list_of_lemmas)

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# we will make pairs of texts to compare and store it into a pandas dataframe
# 
# to know which pair we have we will have also the 'Datum' column as id of input and 'FilePath' as id of output (because there can be outputs for the same time)
# 
# we will merge the texts in the fields of the input (promet.si) to compare with output (student report) without metadata like Datum of course

def process_pair(args):
    input, output_text, output_path = args
    global nlp
    try:
        input_text = ' '.join([input[key] for key in input.keys() if key != 'Datum'])
        tokenized_input = nlp(input_text)
        tokenized_output = nlp(output_text)
        lem_in = lemma_string(tokenized_input)
        lem_out = lemma_string(tokenized_output)
        cos_sim = pairwise_similarity(lem_in, lem_out)
        lem_in_ner = lemma_string_ner(tokenized_input)
        lem_out_ner = lemma_string_ner(tokenized_output)
        # If both ner lemmas are empty (there is no named entity in text), we set cosine similarity to NaN
        if lem_in_ner.strip() and lem_out_ner.strip():
            try:
                cos_sim_ner = pairwise_similarity(lem_in_ner, lem_out_ner)
            except ValueError:
                cos_sim_ner = float('nan')
        else:
            cos_sim_ner = float('nan')
        return {
            'Datum': input.get('Datum', ''),
            'FilePath': output_path,
            'Input': input_text,
            'Output': output_text,
            'CosineSimilarity': cos_sim,
            'CosineSimilarityNER': cos_sim_ner
        }
    except Exception as e:
        return {
            'Datum': input.get('Datum', ''),
            'FilePath': output_path,
            'Input': '',
            'Output': '',
            'CosineSimilarity': float('nan'),
            'CosineSimilarityNER': float('nan')
        }


def make_pairs_pd(json_data, n_jobs=6):
    # Collect all pairs for the whole file
    all_pairs = []
    for item in json_data:
        if 'input' in item and 'output' in item:
            output_path = item['output']['FilePath']
            output_text = item['output']['Content'].split('\n', 1)[1]
            for input in item['input']:
                if 'Datum' in input:
                    all_pairs.append((input, output_text, output_path))
    # Parallelize over all pairs
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=init_pipeline) as executor:
        results = list(executor.map(process_pair, all_pairs))
    df = pd.DataFrame(results)
    return df

def process_and_save(json_data_file_name, file_name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    try:
        logging.info(f"Processing {json_data_file_name}")
        json_data = read_json_file(json_data_file_name)
        df = make_pairs_pd(json_data, n_jobs=6)
        df.to_csv(file_name, index=False)
        logging.info(f"Data saved to {file_name}")
    except Exception as e:
        logging.error(f"Error processing {json_data_file_name}: {e}")

if len(sys.argv) > 1:
    year = sys.argv[1]
else:
    year = '2024'

input_file = f'./Processed/input_output_all_data_{year}_reduced.json'
output_file = f'./Processed/cos_sim_{year}.csv'

process_and_save(input_file, output_file)