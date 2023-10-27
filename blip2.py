import requests
from PIL import Image

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

# here you will want to return the caption of the image that you are dealing with

def getBlip2(image):

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("This is the device: ", device)
    model.to(device)

    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text