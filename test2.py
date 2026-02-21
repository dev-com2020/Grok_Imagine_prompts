import os

from dotenv import load_dotenv
from xai_sdk import Client
import base64

load_dotenv()


client = Client(
    api_key=os.environ["XAI_api_key"],
    timeout=3600, # Override default timeout with longer timeout for reasoning models
)

with open("test.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = client.image.sample(
    prompt="zmień postać na kobietę, weź pod uwagę całą sylwetkę, zmień także ubiór na dresy",
    model="grok-imagine-image",
    image_url=f"data:image/jpeg;base64,{image_data}",
)
print(response.url)