import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_api_key"))

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Nasza "baza danych" dokumentów
dokumenty = [
    "Słońce jest gwiazdą w centrum Układu Słonecznego.",
    "Spaghetti carbonara to klasyczne danie z Rzymu.",
    "Programowanie w Pythonie jest bardzo popularne.",
    "Góry zapewniają piękne widoki i szlaki turystyczne."
]

# Generowanie embeddingów dla wszystkich dokumentów
embeddingi_dokumentow = [get_embedding(doc) for doc in dokumenty]

# Zapytanie użytkownika
zapytanie = "marki niemieckich samochodów?"
embedding_zapytania = get_embedding(zapytanie)

# Obliczanie podobieństwa zapytania do każdego dokumentu
podobieństwa = [cosine_similarity(embedding_zapytania, emb_doc) for emb_doc in embeddingi_dokumentow]

najlepszy_index = np.argmax(podobieństwa)
najlepsze_podobieństwo = podobieństwa[najlepszy_index]

PRÓG = 0.21

print(f"\nZapytanie: {zapytanie}")
print("Wyniki podobieństwa (od najlepszego):")
print("-" * 50)

# Sortujemy wyniki malejąco
posortowane = sorted(enumerate(podobieństwa), key=lambda x: x[1], reverse=True)

for idx, score in posortowane:
    print(f"{score:.4f}  →  {dokumenty[idx]}")

print("-" * 50)

if najlepsze_podobieństwo >= PRÓG:
    print(f"\nOdpowiedź: {dokumenty[najlepszy_index]}")
    print(f"(podobieństwo: {najlepsze_podobieństwo:.4f})")
else:
    print(f"\nBrak odpowiedniej informacji (najwyższe podobieństwo tylko {najlepsze_podobieństwo:.4f} < {PRÓG})")