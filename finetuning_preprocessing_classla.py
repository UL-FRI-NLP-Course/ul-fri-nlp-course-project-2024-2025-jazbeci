# %% [markdown]
# ## WARNING! for this file you need classla. you might get conflicts with tensorflow so i suggest you use a separate python environment for this one
# 
# in this notebook we are tokenizin input text (text from promet.si up to one hour before the reading of traffic news) and the output (the traffic news)
# 
# so if we have traffic news from 1.1.2024 11:30, we will have as input texts all the rows of the excel file (probably web scraped) that are from 1.1.2024 10:30 to 11:30

# %%
import classla
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os


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

def make_pairs_pd(json_data, nlp):
    input_texts = []
    output_texts = []
    datums = []
    output_paths = []
    cosine_similaritys = []
    cosine_similaritys_ner = []
    for item in tqdm(json_data):
        if 'input' in item and 'output' in item:
            inputs = item['input']
            output_path = item['output']['FilePath']
            output_text = item['output']['Content']
            # remove first row of student report when comparing texts
            output_text = output_text.split('\n', 1)[1]
            for input in inputs:
                if 'Datum' in input:
                    datum = input['Datum']
                    input_text = ' '.join([input[key] for key in input.keys() if key != 'Datum'])
                    datums.append(datum)
                    output_paths.append(output_path)
                    input_texts.append(input_text)
                    output_texts.append(output_text)
                    # calculate cosine similarity
                    cosine_similaritys.append(pairwise_similarity(lemma_string(nlp(input_text)), lemma_string(nlp(output_text))))
                    # NER-based similarity with empty check
                    lem_in_ner = lemma_string_ner(nlp(input_text))
                    lem_out_ner = lemma_string_ner(nlp(output_text))
                    if lem_in_ner.strip() and lem_out_ner.strip():
                        try:
                            cosine_similaritys_ner.append(pairwise_similarity(lem_in_ner, lem_out_ner))
                        except ValueError:
                            print('ValueError (ner similarity): ', lem_in_ner, lem_out_ner)
                            cosine_similaritys_ner.append(float('nan'))
                    else:
                        cosine_similaritys_ner.append(float('nan'))
                else:
                    # print('No datum field in input: ', input)
                    continue
                
    # create DataFrame
    df = pd.DataFrame({
            'Datum': datums,
            'FilePath': output_paths,
            'Input': input_texts,
            'Output': output_texts,
            'CosineSimilarity': cosine_similaritys
        })
    return df

# save df to csv
def save_to_csv(data, file_name):
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)


def process_and_save(json_data_file_name, file_name):
    # Initialize the classla pipeline for Slovenian language
    # Only download if not present
    if not os.path.exists(os.path.expanduser('~/.classla')):
        classla.download('sl')
    nlp = classla.Pipeline('sl', processors='tokenize,ner,pos,lemma')
    json_data = read_json_file(json_data_file_name)
    df = make_pairs_pd(json_data, nlp)
    df.to_csv(file_name, index=False)
    print(f"Data saved to {file_name}")

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.submit(process_and_save, './Processed/input_output_all_data_2022_reduced.json', './Processed/cos_sim_2022.csv')
        executor.submit(process_and_save, './Processed/input_output_all_data_2023_reduced.json', './Processed/cos_sim_2023.csv')
        executor.submit(process_and_save, './Processed/input_output_all_data_2024_reduced.json', './Processed/cos_sim_2024.csv')
