import base64
import requests
import re
import os
import openai
import time

# here we will be getting multiple images, and then we will be testing them out
directory_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image"

file_list = os.listdir(directory_path)

index = 1

# for testing with various test images
for filename in file_list:
    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_gpt4.txt", 'a') as file:
        file.write("IMAGE NUMBER: " + str(index) + ", Filename: " + filename + " \n")

    query = ""
    while query != "n":
        query = input("What follow up question do you have? (if no questions enter n): ")

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_gpt4.txt", 'a') as file:
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

            with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_gpt4.txt", 'a') as file:
                file.write("Query time: " + str(elapsed_time) + "s \n")

            # here is how we will get the json string that we will be returning back
            response_string = response.json()["choices"][0]["message"]["content"]

            with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_gpt4.txt", 'a') as file:
                file.write("Answer: " + str(response_string) + "s \n")

            print(response_string)

    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_gpt4.txt", 'a') as file:
        file.write("\n")
    index = index + 1