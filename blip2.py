import requests
from PIL import Image

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

# here you will want to return the caption of the image that you are dealing with

def getBlip2(image):

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, resume_download=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("This is the device: ", device)
    model.to(device)

    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print("First Image: ", generated_text)

    file_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\im2.png"
    image2 = None
    # this is where we will be getting the image object which we will
    # be passing into the summarization method
    try:
        image2 = Image.open(file_path).convert('RGB')
        # Now you can work with the 'image' object
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    inputs2 = processor(image2, return_tensors="pt").to(device, torch.float16)

    generated_ids2 = model.generate(**inputs2, max_new_tokens=50)
    generated_text2 = processor.batch_decode(generated_ids2, skip_special_tokens=True)[0].strip()

    print(generated_text2)

    return generated_text


if __name__ == '__main__':

    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")

    file_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\cross_walk.jpg"
    image = None
    # this is where we will be getting the image object which we will
    # be passing into the summarization method
    try:
        image = Image.open(file_path).convert('RGB')
        # Now you can work with the 'image' object
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    response = getBlip2(image)

    print(response)
