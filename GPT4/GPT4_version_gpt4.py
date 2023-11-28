import base64
import requests
import re
import os
import openai
import time

'''''''''
Definition: this code allows a user to pass in an image to query about and 
            this code calls gpt4 to answer questions about an image
            
Parameters:
- filename: this is the file that we are going to question about 
'''
def call_gtp4(query):

    # OpenAI API Key
    api_key = "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"


    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    # Path to your image
    image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\current.png"

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # here is how we will get the json string that we will be returning back
    response_string = response.json()["choices"][0]["message"]["content"]

    return response_string














