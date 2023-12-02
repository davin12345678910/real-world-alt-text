import base64
import requests
import os
import re
import openai
import time
import json


def get_gpt4_frontload(path):

    with open("bench_mark_frontload_gpt4.txt", 'a') as current_file:
        current_file.write("File: " + path + "\n")

    # OpenAI API Key
    api_key = "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"


    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = """Develop me a valid json that describes the given image. Below are some guidelines to follow and the json structure that is required. 
    
    Please consider:
    Give me only one beginning object for the overall scenery and include for the one json object include information such as:
    Weather 
    Background information, for example are there trees, a sky or sunset shown
    Give me the maximum amount of objects in the image that you can get 
    Children objects are structured the same way as the parent object and themselves can have children objects as well that are structured the same way. For example, a bus can have a person and person can have a shirt  
    For text include any text and numbers you can find on each object 
    Most important things to include in the json: 
    Color information of an object (and include all colors you can find in each object) 
    Actions of an object 
    For children objects, include information about what are object is wearing, what i contains in it if visible and more if possible 
    If there is no text on an object you can give an empty list: []
    Have the objects in the json from left to right 
    For the list of dense captions include information such as the action the objects is taking, what the object color is and any information that relates the current objects with any other objects in the given image
    If there are multiple objects of the same name make multiple objects with names as follow: <object_name>1, <object_name>2, …, <object_name>n
    For possible safety concerns, include information that a blind or low vision person may want to consider such as: moving objects that might impact them, if something might be an obstacle in the path in which they are walking and more
    Lastly give me a json that is within the token limit of 4096 tokens without losing information and maintaining the json structure that is desired 
    Json structure that is required:
    {
       "Objects in image": [
           {
               "Object name": name,
               "Children objects": [],
               "List of Dense captions for object":[dense caption, …, dense caption],
               "Text on object": [text, …, text],
               "Possible safety concerns": [text, …, text]
           },
    
    
           <other objects in image>
       ]
    }"""


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

    json_data = None

    # there are cases where we might hit a rate limit or a call does not work and we will need to retry the call
    # this is why we have a loop, and if there is a failed call we will call the api again
    loop = True
    while loop:
        try:
            start_time = time.time()
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            end_time = time.time()

            elapsed_time = end_time - start_time

            with open("bench_mark_frontload_gpt4.txt", 'a') as file:
                file.write("Front loading runtime: " + str(elapsed_time) + "s \n")

            # here is how we will get the json string that we will be returning back
            json_string = response.json()["choices"][0]["message"]["content"]

            json_data_pre = json_string[json_string.index("{") : len(json_string) - 3]

            json_data = json_data_pre[0 : json_data_pre.rfind("}") + 1]

            # load the json data into a file
            json_current = json.loads(json_data)

            # here we will opening the file
            with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\GPT4Frontload\\json_gpt4.json", 'w') as file:
                json.dump(json_current, file, indent=4)

            loop = False
        except Exception as e:
            print("Error: ", e)

    return json_data
