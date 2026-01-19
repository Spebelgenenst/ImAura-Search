import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "an image of a cat"

inputs = processor(text=text, images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# multi-modal text embedding
text_embeds = outputs.text_embeds

# multi-modal image embedding
image_embeds = outputs.image_embeds