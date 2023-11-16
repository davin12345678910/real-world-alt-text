import base64
import requests
import os
import re
import openai
import time

# we will loop through the images and then we will output each of the responses
# here we will be getting multiple images, and then we will be testing them out
directory_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text-GPT4\\test-image"

file_list = os.listdir(directory_path)

index = 1

# for testing with various test images
# for filename in file_list:
    # print("CURRENT IMAGE: ", filename)

# OpenAI API Key
api_key = "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"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Path to your image
image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text-GPT4\\test-image\\king_county_buses.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

prompt = "Develop me a json that describes the given image. Below are some guidelines to follow and the json structure that is required.\n" \
"Please consider: \n" \
"Give me only one beginning object for the overall scenery and include for the one json object include information such as: \n" \
"- Weather \n" \
"- Background information, for example are there trees, a sky or sunset shown \n" \
"- Give me all the objects that you can find in the image including objects in the background that might not be close by \n" \
"- Children objects are structured the same way as the parent object and themselves can have children objects as well that are structured the same way. For example, a bus can have a person and person can have a shirt \n" \
"- Most important things to include in the json: \n" \
"    - Any dangerous obstacles or objects please include them in the json \n" \
"    - Color information of an object \n" \
"    - Actions of an object \n" \
"- For children objects, include information about what are object is wearing, what i contains in it if visible and more if possible \n" \
"- If there is no text on an object you can give an empty list: [] \n" \
"- Have the objects in the json from left to right \n" \
"- For the list of dense captions include information such as the action the objects is taking, what the object color is and any information that relates the current objects with any other objects in the given image \n" \
"- If there are multiple objects of the same name make multiple objects with names as follow: <object_name>1, <object_name>2, …, <object_name>n \n" \
"- For possible safety concerns, include information that a blind or low vision person may want to consider such as: moving objects that might impact them, if something might be an obstacle in the path in which they are walking and more \n" \
"- Lastly give me a json that is within the token limit of 4096 tokens without losing information and maintaining the json structure that is desired \n" \
"Json structure that is required: \n" \
"{\n" \
"   \"Objects in image\": [ \n" \
"       { \n" \
"           \"Object name\": name, \n" \
"           \"Children objects\": [], \n" \
"           \"List of Dense captions for object\":[dense caption, …, dense caption], \n" \
"           \"Text on object\": [text, …, text], \n" \
"           \"Possible safety concerns\": [text, …, text], \n" \
"       }, \n" \
"       <other objects in image> \n " \
"   ] \n" \
"} \n" \


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
try:
    start_time = time.time()
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    end_time = time.time()

    elapsed_time = end_time - start_time

    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark2.txt", 'a') as file:
        file.write("Front loading runtime: " + str(elapsed_time) + "s \n")



    # print("GPT JSON: ", response.json())

    # here is how we will get the json string that we will be returning back
    json_string = response.json()["choices"][0]["message"]["content"]

    # print("JSON_STRING: \n ", json_string)

    json_data_pre = json_string[json_string.index("{") : len(json_string) - 3]

    json_data = json_data_pre[0 : json_data_pre.rfind("}") + 1]
except Exception as e:
    print(e)

print("JSON DATA: ", json_data)

# here we are going to be querying gpt4 with chat completions
query = ""
while query != "n":
    query = input("What follow up question do you have? (if no questions enter n): ")

    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark2.txt", 'a') as file:
        file.write("Question asked: " + str(query) + "s \n")

    if query != "n":
        openai.api_key = "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"

        prompt = json_data + "\n Given this json, answer the current question while making sure of the following things: \n" \
                             "do not mention" \
                             " coordinates or bounding boxes, do not give me information that is not in the json " \
                             "or history, if you do not have information about something from the json or hierachy " \
                             "state that you do not have information about it. Please answer the following question as if you were " \
                             "talking to a person who is blind or has low vision: " + query \

        start_time = time.time()
        # here we will be building the string that we will put into content
        gpt4_results = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        end_time = time.time()

        elapsed_time = end_time - start_time

        response = gpt4_results.choices[0]["message"]["content"]

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark2.txt", 'a') as file:
            file.write("Question asked: " + str(query) + "s \n")

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark2.txt", 'a') as file:
            file.write("GPT4 answer: " + str(response) + "s \n")

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark2.txt", 'a') as file:
            file.write("Query runtime: " + str(elapsed_time) + "s \n")

with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark2.txt", 'a') as file:
    file.write("\n")


# WIST: JULY, just the system with the conversation and the gesture, just from the glasses and a comparison with seeing ai. How does the hjand gesture compare to the seeing ai with the touch based approach
# Full paper with the real-time stuff with KAI
# we are really going to have to focus on the real-time stuff