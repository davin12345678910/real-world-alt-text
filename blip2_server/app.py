from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import io

app = FastAPI()

# here we will initialize the pretrained parts first
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

''''''''''
Definition:
this endpoint allows a user to pass in an image an a prompt

Parameters:
image - this is the image that we will be giving an answer for 
text - this is the prompt that the user passed in to query an image 

Return:
a string of the answer to a persons given image and query 
'''
@app.post('/blip2_predict')
def blip2_predict(image: UploadFile = File(...), text: str = Form(...)):

    # this is the image that the user passed in
    image_data = image.file.read()
    image = Image.open(io.BytesIO(image_data))

    # here we run the model to get a result
    inputs = processor(image, text=text, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=50)

    # here we will return the output of the text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


#syncronus version
'''''''''
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import io

app = FastAPI()

# Pretrained parts initialization
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

def process_image_and_predict(image_data, text):
    # Process and predict in a synchronous manner
    image = Image.open(io.BytesIO(image_data))
    inputs = processor(image, text=text, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

@app.post('/blip2_predict')
async def blip2_predict(background_tasks: BackgroundTasks, image: UploadFile = File(...), text: str = Form(...)):
    # Read file asynchronously
    image_data = await image.read()

    # Offload processing to background
    background_tasks.add_task(process_image_and_predict, image_data, text)

    return {"message": "Processing started. The result will be handled in the background."}
'''

