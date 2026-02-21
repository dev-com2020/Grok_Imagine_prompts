import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_api_key"))

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": """
        Twoim zadaniem jest klasyfikacja sentymentu ocen filmów.
        """ },
        # Few-shot
        {"role": "user", "content": "Film do bani! Nigdy więcej!"},
        {"role": "assistant", "content": "Negatywny" },

        {"role": "user", "content": "Bardzo ciekawy film! Chętnie obejrze ponownie."},
        {"role": "assistant", "content": "Pozytywny" },

        {"role": "user", "content": """
        Okropny ten film!!!
        """}
    ],
    temperature=0.5,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.1
)
print(response.choices[0].message.content)
print("Ilość tokenów", response.usage)