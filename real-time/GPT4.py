import base64
import requests
import re


def get_gpt4(path, prompt, key):
  # OpenAI API Key
  api_key = key

  # Function to encode the image
  def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

  # Path to your image
  image_path = path

  # Getting the base64 string
  base64_image = encode_image(image_path)

  # headers for content type and openai token
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  # information that we will be passing to the model
  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": prompt
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

  # response json from ogpt4
  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  print(response.json())

  # here is how we will get the json string that we will be returning back
  json_string = response.json()["choices"][0]["message"]["content"]

  return json_string