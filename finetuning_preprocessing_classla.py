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

# %%
classla.download('sl')

# %%
nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma')

# %%
test_output = 'Prometne informacije       01. 01. 2022  \t   11.30          2. program \n\nPodatki o prometu.\n\nNa gorenjski avtocesti proti Ljubljani je zaradi gorečega vozila zaprt vozni pas med priključkoma Brezje in Podtabor.   \n\nDanes do 21-ih velja prepoved prometa tovornih vozil, težjih od 7 ton in pol.\n'
test_input = 'Vreme Ponekod v osrednji Sloveniji megla v pasovih zmanjšuje vidljivost. Omejitve za tovorna vozila Po Sloveniji velja med prazniki omejitev za tovorna vozila z največjo dovoljeno maso nad 7,5 ton: - danes do 22. ure; - v nedeljo, 2. januarja, od 8. do 22. ure. Od 30. decembra je v veljavi sprememba omejitve za tovorna vozila nad 7,5 ton. Več.'
# lets remove first row of student report when comparing texts
test_output = test_output.split('\n', 1)[1]
doc_out = nlp(test_output)
doc_in = nlp(test_input)
doc_out

# %%
def lemma_string(doc):
  list_of_lemmas = [word.lemma for t in doc.iter_tokens() for word in t.words]
  odstrani_locila = [x for x in list_of_lemmas if x not in [',','.', ':', '-', ';']]
  list_to_string = ' '.join(odstrani_locila)
  return list_to_string

lemmas_out = lemma_string(doc_out)
print(lemmas_out)
lemmas_in = lemma_string(doc_in)
print(lemmas_in)

# %%
def pairwise_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

print(pairwise_similarity(lemmas_out, lemmas_in))

# %%
nlp_ner = classla.Pipeline('sl', processors='tokenize,ner,pos,lemma')

# %%
doc_out_ner = nlp_ner(test_output)
doc_in_ner = nlp_ner(test_input)
doc_out_ner

# %%
for sent in doc_in_ner.sentences:
    for token in sent.tokens:
        for word in token.words:
            print(word.to_dict())

# %%
def lemma_string_ner(doc):
    list_of_lemmas = []
    for sentence in doc.sentences:
        for t, w in zip(sentence.tokens, sentence.words):
            if(t.ner != 'O'):
                list_of_lemmas.append(w.lemma)
    return ' '.join(list_of_lemmas)

# Example usage:
lemmas_out_ner = lemma_string_ner(doc_out_ner)
print(lemmas_out_ner)
lemmas_in_ner = lemma_string_ner(doc_in_ner)
print(lemmas_in_ner)

# %%
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

data_2022 = read_json_file('./Processed/input_output_all_data_2022_reduced.json')
data_2023 = read_json_file('./Processed/input_output_all_data_2023_reduced.json')
data_2024 = read_json_file('./Processed/input_output_all_data_2024_reduced.json')
data_2022[0]

# %%
#check if there is any input that does not have fields that are not A1, B1, C1, A2, B2 or C2
def check_fields(data):
    count = 0
    for item in data:
        if 'input' in item and 'output' in item:
            inputs = item['input']
            output_text = item['output']
            for input in inputs:
                is_in = 0
                is_not = 0
                for key in input.keys():
                    if key in ['A1', 'B1', 'C1', 'A2', 'B2', 'C2']:
                        is_in += 1
                    else:
                        is_not += 1
                if is_in == 0 and is_not > 0:
                    count += 1
    print(count)
                        
                
check_fields(data_2022)
check_fields(data_2023)
check_fields(data_2024)

# %% [markdown]
# we will make pairs of texts to compare and store it into a pandas dataframe
# 
# to know which pair we have we will have also the 'Datum' column as id of input and 'FilePath' as id of output (because there can be outputs for the same time)
# 
# we will merge the texts in the fields of the input (promet.si) to compare with output (student report) without metadata like Datum of course

# %%
def make_pairs_pd(json_data):
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
                    lem_in_ner = lemma_string_ner(nlp_ner(input_text))
                    lem_out_ner = lemma_string_ner(nlp_ner(output_text))
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


# %%
df_2022 = make_pairs_pd(data_2022)
tqdm(save_to_csv(data_2022, 'Processed/cos_sim_2022.csv'))

# %%


# %%
df_2022 = make_pairs_pd(data_2022)
save_to_csv(data_2023, 'Processed/cos_sim_2023.csv')


# %%
df_2022 = make_pairs_pd(data_2022)
save_to_csv(data_2024, 'Processed/cos_sim_2024.csv')


