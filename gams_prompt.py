
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import json
import random

from transformers import pipeline

model_id = "cjvt/GaMS-9B-Instruct"

pline = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",  # Automatically distribute across GPUs and CPU
)

# Load the JSON data from the file
file_path = '/d/hpc/projects/onj_fri/jazbeci/ul-fri-nlp-course-project-2024-2025-jazbeci/RTVSlo/Processed_Data_First_10000_non_nan.json'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

indices = [300, 1100, 9530]

b1_contents = [data[i].get('B1', '') for i in indices]

for i, b1 in enumerate(b1_contents, 1):
    print(f"Processing B1 content {i}:\n{b1}\n")
    
    message = f"Generiraj prometno novico v enem odstavku z uporabo naslednjih podatkov:\n\n{b1}\n\nFormatiraj izhod na naslednji način: vsak dogodek naj bo zapisan v obliki 'Razlog + cesta in smer + posledica in odsek'."
    message = [{"role": "user", "content": message}]
    response = pline(message, max_new_tokens=200)

    print("Model response:", response[0]["generated_text"])
    print("*************")
