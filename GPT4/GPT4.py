import base64
import requests
import re


def get_gpt4(prompt):
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

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  # print("GPT JSON: ", response.json())

  # here is how we will get the json string that we will be returning back
  json_string = response.json()["choices"][0]["message"]["content"]
  json_final_string = json_string[7 : len(json_string) - 3]

  return json_final_string