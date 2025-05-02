
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

prompt = """
Sestavi prometno poročilo za 11.30 v odstavkih po naslednjih pravilih:

1. FORMAT:
Prometne informacije       17. 01. 2022       11.30              2. program

Podatki o prometu.

[VSEBINA]

2. PRAVILA ZA VSEBINO

    Kratki, aktivni stavki (npr. "Na avtocesti A1 je zaradi nesreče zastoj.").

    Vsak dogodek v svoji vrstici.

    Struktura: Lokacija → Razlog → Posledice (npr. "Na AC Maribor–Ljubljana je zaradi dela zaprt desni pas, promet je počasen.").

3. VRSTNI RED

(Natančno upoštevaj to zaporedje!)

    Voznik v napačno smer

    Zaprta avtocesta

    Nesreča z zastojem na avtocesti

    Zastoji zaradi del na avtocesti (krajši zastoj + povečana nevarnost naletov)

    Zaprta glavna/regionalna cesta (zaradi nesreče)

    Nesreče na drugih cestah

    Pokvarjena vozila (z zaprtjem pasu)

    Žival na vozišču

    Predmet/razsut tovor na avtocesti

    Dela na avtocesti z večjo nevarnostjo (zaprtje pasov, predori)

    Zastoj pred Karavankami/mejnimi prehodi (VEDNO NA KONCU!)

4. ODPOVEDI
    Če se stanje razreši, dodaj odpoved prometne informacije.
    Primer formulacije za odpoved prometne informacije o vozniku v napačni smeri:
    Promet na pomurski avtocesti iz smeri Dragučove proti Pernici ni več ogrožen zaradi voznika, ki je vozil po napačni polovici avtoceste. 

5. VREMENSKE INFORMACIJE

    Na koncu dodaj meglo, močan veter, zaprte planinske ceste ipd., če niso neposredna prometna ovira.

6. PREVERJANJE

    Pred oddajo preveri, ali so vse kategorije obdelane v pravilnem vrstnem redu (npr. ali je prometna informacija o zastoju pred Karavankah na koncu).

    Vsaka prometna informacija mora biti v svoji vrstici

PODATKI ZA OBDELAVO:
Glej podatke od 11.00 naprej, prejšnje uporabi le za preverjanje odpovedi
[
    {
        "Datum": "2022-01-17 10:32:15",
        "ContentVremeSLO": "Ponekod po državi megla zmanjšuje vidljivost.Cesta čez prelaz Vršič je prevozna samo za osebna vozila.",
        "TitleDeloNaCestiSLO": "Dela",
        "ContentDeloNaCestiSLO": "Cesta Ilirska Bistrica - Podgrad je zaprta pri odcepu za Podbeže do 24. januarja. Obvoz za vozila do 7,5 tone in avtobuse je po cesti Podgrad - Obrov - Pregarje - Harije. Za vozila nad 7,5 tone pa po primorski avtocesti. ",
        "ContentNesreceSLO": "Na primorski avtocesti med Vrhniko in Brezovico proti Ljubljani oviran promet na odstavnem pasu.",
        "ContentOvireSLO": "Na štajerski avtocesti med priključkoma Celje zahod in Žalec proti Ljubljani predmet na prehitevalnem pasu.",
        "ContentMednarodneInformacijeSLO": "Čakalna doba je na Gruškovju in Obrežju.",
        "TitleSplosnoSLO": "Odstranjevanje nevarnega predmeta"
    },
    {
        "Datum": "2022-01-17 10:39:58",
        "ContentVremeSLO": "Ponekod po državi megla zmanjšuje vidljivost.Cesta čez prelaz Vršič je prevozna samo za osebna vozila.",
        "TitleDeloNaCestiSLO": "Dela",
        "ContentDeloNaCestiSLO": "Cesta Ilirska Bistrica - Podgrad je zaprta pri odcepu za Podbeže do 24. januarja. Obvoz za vozila do 7,5 tone in avtobuse je po cesti Podgrad - Obrov - Pregarje - Harije. Za vozila nad 7,5 tone pa po primorski avtocesti. ",
        "ContentNesreceSLO": "Na primorski avtocesti med Vrhniko in Brezovico proti Ljubljani oviran promet na odstavnem pasu.",
        "ContentOvireSLO": "Zaradi predmeta je oviran promet na dolenjski avtocesti med Mirno Pečjo in in Novim mestom.",
        "ContentMednarodneInformacijeSLO": "Čakalna doba je na Gruškovju in Obrežju.",
        "TitleSplosnoSLO": "Odstranjevanje nevarnega predmeta"
    },
    {
        "Datum": "2022-01-17 10:52:26",
        "ContentVremeSLO": "Ponekod po državi megla zmanjšuje vidljivost.Cesta čez prelaz Vršič je prevozna samo za osebna vozila.",
        "TitleDeloNaCestiSLO": "Dela",
        "ContentDeloNaCestiSLO": "Cesta Ilirska Bistrica - Podgrad je zaprta pri odcepu za Podbeže do 24. januarja. Obvoz za vozila do 7,5 tone in avtobuse je po cesti Podgrad - Obrov - Pregarje - Harije. Za vozila nad 7,5 tone pa po primorski avtocesti. ",
        "ContentNesreceSLO": "Na primorski avtocesti med Vrhniko in Brezovico proti Ljubljani oviran promet na odstavnem pasu.",
        "ContentOvireSLO": "Zaradi predmeta je oviran promet na dolenjski avtocesti med Mirno Pečjo in in Novim mestom.",
        "ContentMednarodneInformacijeSLO": "Čakalna doba je na Gruškovju in Obrežju.",
        "TitleSplosnoSLO": "Odstranjevanje nevarnega predmeta"
    },
    {
        "Datum": "2022-01-17 11:01:43",
        "ContentVremeSLO": "Ponekod po državi megla zmanjšuje vidljivost.Cesta čez prelaz Vršič je prevozna samo za osebna vozila.",
        "TitleDeloNaCestiSLO": "Dela",
        "ContentDeloNaCestiSLO": "Cesta Ilirska Bistrica - Podgrad je zaprta pri odcepu za Podbeže do 24. januarja. Obvoz za vozila do 7,5 tone in avtobuse je po cesti Podgrad - Obrov - Pregarje - Harije. Za vozila nad 7,5 tone pa po primorski avtocesti. ",
        "ContentMednarodneInformacijeSLO": "Čakalna doba je na Gruškovju in Obrežju.",
        "TitleSplosnoSLO": "Odstranjevanje nevarnega predmeta"
    },
    {
        "Datum": "2022-01-17 11:03:16",
        "ContentVremeSLO": "Ponekod po državi megla zmanjšuje vidljivost.Cesta čez prelaz Vršič je prevozna samo za osebna vozila.",
        "TitleDeloNaCestiSLO": "Dela",
        "ContentDeloNaCestiSLO": "Cesta Ilirska Bistrica - Podgrad je zaprta pri odcepu za Podbeže do 24. januarja. Obvoz za vozila do 7,5 tone in avtobuse je po cesti Podgrad - Obrov - Pregarje - Harije. Za vozila nad 7,5 tone pa po primorski avtocesti. ",
        "ContentMednarodneInformacijeSLO": "Čakalna doba je na Gruškovju in Obrežju.",
        "ContentZastojiSLO": "Na gorenjski avtocesti je zastoj tovornih vozil pred predorom Karavanke proti Avstriji, približno 1,5 kilometra.",
        "TitleSplosnoSLO": "Odstranjevanje nevarnega predmeta"
    },
    {
        "Datum": "2022-01-17 11:16:51",
        "ContentVremeSLO": "Ponekod po državi megla zmanjšuje vidljivost.Cesta čez prelaz Vršič je prevozna samo za osebna vozila.",
        "TitleDeloNaCestiSLO": "Dela",
        "ContentDeloNaCestiSLO": "Cesta Ilirska Bistrica - Podgrad je zaprta pri odcepu za Podbeže do 24. januarja. Obvoz za vozila do 7,5 tone in avtobuse je po cesti Podgrad - Obrov - Pregarje - Harije. Za vozila nad 7,5 tone pa po primorski avtocesti. ",
        "ContentOvireSLO": "Predmet je na dolenjski avtocesti pred izvozom Mirna Peč proti Novemu mestu.",
        "ContentMednarodneInformacijeSLO": "Čakalna doba je na Gruškovju in Obrežju.",
        "ContentZastojiSLO": "Na gorenjski avtocesti je zastoj tovornih vozil pred predorom Karavanke proti Avstriji, približno 1,5 kilometra.",
        "TitleSplosnoSLO": "Odstranjevanje nevarnega predmeta"
    },
    {
        "Datum": "2022-01-17 11:19:50",
        "ContentVremeSLO": "Ponekod po državi megla zmanjšuje vidljivost.Cesta čez prelaz Vršič je prevozna samo za osebna vozila.",
        "TitleDeloNaCestiSLO": "Dela",
        "ContentDeloNaCestiSLO": "Cesta Ilirska Bistrica - Podgrad je zaprta pri odcepu za Podbeže do 24. januarja. Obvoz za vozila do 7,5 tone in avtobuse je po cesti Podgrad - Obrov - Pregarje - Harije. Za vozila nad 7,5 tone pa po primorski avtocesti. ",
        "ContentOvireSLO": "Predmet je na dolenjski avtocesti pred izvozom Mirna Peč proti Novemu mestu.",
        "ContentMednarodneInformacijeSLO": "Čakalna doba je na Obrežju, Dobovcu in Gruškovju.",
        "ContentZastojiSLO": "Na gorenjski avtocesti je zastoj tovornih vozil pred predorom Karavanke proti Avstriji, približno 1,5 kilometra.",
        "TitleSplosnoSLO": "Odstranjevanje nevarnega predmeta"
    },
    {
        "Datum": "2022-01-17 11:20:33",
        "ContentVremeSLO": "Ponekod po državi megla zmanjšuje vidljivost.Cesta čez prelaz Vršič je prevozna samo za osebna vozila.",
        "TitleDeloNaCestiSLO": "Dela",
        "ContentDeloNaCestiSLO": "Cesta Ilirska Bistrica - Podgrad je zaprta pri odcepu za Podbeže do 24. januarja. Obvoz za vozila do 7,5 tone in avtobuse je po cesti Podgrad - Obrov - Pregarje - Harije. Za vozila nad 7,5 tone pa po primorski avtocesti. ",
        "ContentOvireSLO": "Predmet je na dolenjski avtocesti pred izvozom Mirna Peč proti Novemu mestu.",
        "ContentMednarodneInformacijeSLO": "Čakalna doba je na Obrežju, Dobovcu in Gruškovju.",
        "ContentZastojiSLO": "Na gorenjski avtocesti je zastoj tovornih vozil pred predorom Karavanke proti Avstriji, približno 1,5 kilometra."
    },
]

PREDHODNO POROČILO:

Prometne informacije       17. 01. 2022       11.00              2. program 

Podatki o prometu.

Po podatkih voznikov je ponekod po državi močno zmanjšana vidljivost zaradi megle. Voznikom svetujemo, naj prilagodijo hitrost vožnje, povečajo varnostno razdaljo ter po potrebi vklopijo meglenke.

Pred predorom Karavanke je kilometer dolg zastoj tovornih vozil proti Avstriji.

Na mejnih prehodih Gruškovje in Petišovci vozniki tovornih vozil na vstop v Slovenijo čakajo 1 uro. 
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

