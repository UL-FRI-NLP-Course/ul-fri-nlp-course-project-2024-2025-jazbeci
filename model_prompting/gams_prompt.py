
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

input_data = "Na štajerski avtocesti je zaradi okvare vozila oviran promet med predorom Podmilj in priključkom Blagovica proti Ljubljani. Promet je zaradi jutranje prometne konice povečan na cestah, ki vodijo v mestna središča ter na mestnih obvoznicah. Zaradi močnega vetra je v Luki Koper zaprt kontejnerski terminal. Primorska avtocesta je zaradi močnega vetra zaprta med priključkoma Kastelec in Kozina proti Ljubljani. Zaprt je tudi uvoz Kastelec proti Ljubljani. Obvoz je po regionalni cesti. BURJA Prepovedan promet za počitniške prikolice, hladilnike, vozila s ponjavami in avtobuse, 3. stopnja; - na vipavski hitri cesti med razcepom Nanos in priključkom Selo; - na regionalni cesti Vipava - Ajdovščina. Prepovedan promet za počitniške prikolice, hladilnike in vozila s ponjavami do 8 ton, 1. stopnja; - na regionalni cesti Razdrto - Podnanos in Selo - Ajševica - Vogrsko. Cesta Smednik - Kostanjevica je zaprta zaradi poplavljenega vozišča pri Zameškem. Cesta Maribor - Pernica je zaradi plazu zaprta med Malečnikom in krožiščem pri avtocestnem priključku Pernica. Več o delovnih zaporah v prometni napovedi."


prompt = f"""
Sestavi prometno novico v odstavkih iz vhodnih podatkov po naslednjih pravilih:
VSAKA PROMETNA INFORMACIJA JE ZAPISANA V SVOJI VRSTICI
PREBERI VHODNE PODATKE IN JIH UREDI PO SLEDEČEM VRSTNEM REDU (ČE SE V PODATKIH POJAVIJO):
Voznik v napačno smer
Zaprta avtocesta
Prometna nesreča na avtocesti 
Zastoji zaradi del na avtocesti 
Zaprta glavna/regionalna cesta 
Prometne nesreče na drugih cestah
Pokvarjena vozila
Žival na vozišču
Predmet ali razsut tovor na avtocesti
Dela na avtocesti z večjo nevarnostjo
Zastoj pred Karavankami ali mejnimi prehodi 

VSAK PODATEK NAPIŠI V SVOJO VRSTICO V SLEDEČI OBLIKI: 
Lokacija razlog posledice 
(npr. "Na AC Maribor-Ljubljana je zaradi dela zaprt desni pas, promet je počasen.").

NOVICO SESTAVI V ČIM BOLJ NARAVNEM JEZIKU, SAJ BO PREBRANA NA RADIU. NE SPREMINJAJ DEJSTEV ZNOTRAJ PODATKOV. 

PRIMER:
ČE SO VHODNI PODATKI:
Cesta Slovenj Gradec - Ravne na Koroškem je zaprta med Slovenj Gradcem in Starim trgom. Cesta čez prelaz Vršič je prevozna za osebna vozila z verigami. Cesta Podtabor - Ljubelj je zaprta pri predorih v Tržiču zaradi sanacije zemeljskega plazu. Zaprt je tudi prelaz Ljubelj. Več o delovnih zaporah v prometni napovedi.

IZPIŠEŠ PROMETNO NOVICO:

Zaradi prometne nesreče je zaprta regionalna cesta Slovenj Gradec-Kotlje-Ravne na Koroškem med Slovenj Gradcem in Starim trgom.   

Zaradi zemeljskega plazu je cesta Podtabor-Ljubelj zaprta pri predorih v Tržiču. Zaprt je tudi prelaz Ljubelj.

Cesta čez Vršič je prevozna le za osebna vozila z verigami.

VHODNI PODATKI SO:
{input_data}
"""

# Example of response generation
message = [{"role": "user", "content": prompt}]
response = pline(message, max_new_tokens=512)
try:
    print("Model's response:", response[0]["generated_text"][-1]["content"])
except KeyError:
    print("Model's response:", response[0]["generated_text"])
except Exception as e:
    # Print the full response
    print("Model's response:", response)
    print("Error:", e)

