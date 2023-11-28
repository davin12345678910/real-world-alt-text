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
def call_gtp4(filename):
    with open("/bench_mark_gpt4.txt", 'a') as file:
        file.write("Filename: " + filename + " \n")

    query = ""
    while query != "n":
        query = input("What follow up question do you have? (if no questions enter n): ")

        with open("/bench_mark_gpt4.txt", 'a') as file:
            file.write("Question asked: " + str(query) + "s \n")

        if query != "n":
            # OpenAI API Key
            api_key = "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"


            # Function to encode the image
            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')


            # Path to your image
            image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\" + filename

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

            start_time = time.time()
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            end_time = time.time()

            elapsed_time = end_time - start_time

            with open("/bench_mark_gpt4.txt", 'a') as file:
                file.write("Query time: " + str(elapsed_time) + "s \n")

            # here is how we will get the json string that we will be returning back
            response_string = response.json()["choices"][0]["message"]["content"]

            with open("/bench_mark_gpt4.txt", 'a') as file:
                file.write("Answer: " + str(response_string) + "s \n")

    with open("/bench_mark_gpt4.txt", 'a') as file:
        file.write("\n")


if __name__=='__main__':

    # here we will be testing multiple images with the gpt4 system
    directory_path = "/test-image"
    file_list = os.listdir(directory_path)
    for filename in file_list:
        call_gtp4(filename)

















# benchmarking code:
'''''''''
    with open("/bench_mark_gpt4.txt", 'a') as file:
        file.write("Filename: " + filename + " \n")

    query = ""
    while query != "n":
        query = input("What follow up question do you have? (if no questions enter n): ")

        with open("/bench_mark_gpt4.txt", 'a') as file:
            file.write("Question asked: " + str(query) + "s \n")

        if query != "n":
            # OpenAI API Key
            api_key = "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"


            # Function to encode the image
            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')


            # Path to your image
            image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\" + filename

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

            start_time = time.time()
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            end_time = time.time()

            elapsed_time = end_time - start_time

            with open("/bench_mark_gpt4.txt", 'a') as file:
                file.write("Query time: " + str(elapsed_time) + "s \n")

            # here is how we will get the json string that we will be returning back
            response_string = response.json()["choices"][0]["message"]["content"]

            with open("/bench_mark_gpt4.txt", 'a') as file:
                file.write("Answer: " + str(response_string) + "s \n")

    with open("/bench_mark_gpt4.txt", 'a') as file:
        file.write("\n")
'''
