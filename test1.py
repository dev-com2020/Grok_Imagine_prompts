import os

import requests
from dotenv import load_dotenv
from xai_sdk import Client

load_dotenv()


client = Client(
    api_key=os.environ["XAI_api_key"],
    timeout=3600, # Override default timeout with longer timeout for reasoning models
)

response = client.image.sample(
    prompt="mario który pali marihuane, w tle niekompletnie ubrana księżniczna peach",
    model="grok-imagine-image",
    aspect_ratio="4:3"
)

url = response.url
img_data = requests.get(url).content
with open("test.jpg", "wb") as f:
    f.write(img_data)

# pip install python - dotenv