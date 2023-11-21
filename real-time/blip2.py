from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

import requests
from PIL import Image
import os
import time


def test_blip2():
    directory_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\cropped_imgs"

    files_and_directories = os.listdir(directory_path)

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for file in files_and_directories:

        image = Image.open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\cropped_imgs\\" + file).convert('RGB')
        start_time = time.time()
        prompt = "give me information about this image"
        inputs = processor(image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        end_time = time.time()

        print("Elapsed time: ", round(float(end_time - start_time), 4))
        print("Response: ", generated_text)


if __name__ == '__main__':
    test_blip2()