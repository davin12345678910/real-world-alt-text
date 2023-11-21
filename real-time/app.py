from fastapi import FastAPI, UploadFile, File
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import io

app = FastAPI()

# here we will initialize the pretrained parts first
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


@app.post('/blip2_predict')
async def blip2_predict(image: UploadFile = File(...)):
    # this is the image that the user passed in
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    # print("Processed image!")
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # print("Response: ", generated_text)

    return {"text": generated_text}











'''''''''''
from flask import Flask, request
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import io

app = Flask(__name__)

# here we will initialize the pretrained parts first
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


@app.route('/blip2_predict', methods=['POST'])
def blip2_predict():
    # this is the image that the user passed in
    image_data = request.files.get("image")
    image = Image.open(io.BytesIO(image_data.read()))
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    print("Processed image!")
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print("Response: ", generated_text)

    return generated_text, 200
'''