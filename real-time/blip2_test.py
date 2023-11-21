from PIL import Image
import requests
from transformers import AutoProcessor, Blip2Processor, Blip2ForConditionalGeneration
import torch
import requests
from PIL import Image
import os


def test_blip2(processor, model, path, queue):
    # print("WENT INTO: test_blip2!")

    # processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    image = Image.open(path).convert('RGB')
    # prompt = "this is an image of"

    # print("Going to process image")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    # print("Processed image!")
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # print(generated_text)




'''''''''''
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt= 'Describe me this image'

inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=2048)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("Response: ", generated_text)
'''