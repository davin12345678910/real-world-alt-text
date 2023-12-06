from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import io
import asyncio

# here wee will be using the FastAPI in order to get a concurrent server for our API
app = FastAPI()

# Initialize the pretrained models
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

# here we will be making sure that we are using the GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)


'''''''''
Defintion: this is the endpoint that we will be using in order to get results from the blip2 model

Parameters:
image: this is the image that we will be analyzing
text: this is the text that we will be using as our query for the given image 

Returns:
This method returns the prediction from the blip2 model 

Note: this system should be able to handle multiple request in parallel but the max amount of 
concurrent calls should not be surpassed given that we are not going to get enough objects that 
we can't handle it on a GeForce RTX 4080
'''
@app.post('/blip2_predict')
async def blip2_predict(image: UploadFile = File(...), text: str = Form(...)):

    # this is the image data that we will be reading in from the user
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))

    # this is how we will be structuring our input into the blip2 model
    inputs = processor(image, text=text, return_tensors="pt").to(device, torch.float16)

    # Run the model in a background thread
    # this will allow other runs to the model to not be blocked off
    generated_ids = await asyncio.to_thread(
        model.generate,
        **inputs,
        max_new_tokens=50
    )

    # here we will be returning the results from the model
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text
