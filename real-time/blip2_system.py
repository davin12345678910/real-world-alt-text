from ultralytics import YOLO

import requests
# from transformers import AutoProcessor, Blip2Processor, Blip2ForConditionalGeneration
import torch
# import requests
from PIL import Image
import os
import multiprocessing
# import blip2_test
import blip2_endpoint
import openai
import time
# import replicate
from multiprocessing import Pool, Manager
import GPT4
import JsonParser

import json

def get_blip2_response(bbox, path, queue, name):

    response = blip2_endpoint.get_blip2(path, name)
    bbox_formatted = [[bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]]]
    queue.put({"name" : name, "bbox": bbox_formatted, "description" : response})


def real_time_test(path, pool, queue, query, model):

    start_time = time.time()


    results = model(
        [path])

    end_time3 = time.time()

    print("Elapsed_time yolo: ", (end_time3 - start_time))

    boxes = None
    classes = None

    for key in results:
        boxes = key.boxes.xyxy.tolist()
        classes = key.boxes.cls.tolist()

    idToObject = {0: "person",
                  1: "Bicycle",
                  2: "car",
                  3: "Motorcycle",
                  4: "Airplace",
                  5: "Bus",
                  6: "Train",
                  7: "Truck",
                  8: "Boat",
                  9: "Traffic Light",
                  10: "Fire Hydrant",
                  11: "Stop sign",
                  12: "Parking Meter",
                  13: "Bench",
                  14: "Bird",
                  15: "Cat",
                  16: "Dog",
                  17: "Horse",
                  18: "Sheep",
                  19: "Cow",
                  20: "Elephant",
                  21: "Bear",
                  22: "Zebra",
                  23: "Giraffe",
                  24: "Backpack",
                  25: "Umbrella",
                  26: "Handbag",
                  27: "Tie",
                  28: "Suitcase",
                  29: "Frisbee",
                  30: "Skis",
                  31: "Snowboard",
                  32: "Sports ball",
                  33: "Kite",
                  34: "Baseball bat",
                  35: "Baseball glove",
                  36: "Skateboard",
                  37: "Surfboard",
                  38: "Tennis racket",
                  39: "Bottle",
                  40: "Wine glass",
                  41: "Cup",
                  42: "Fork",
                  43: "Knife",
                  44: "Spoon",
                  45: "Bowl",
                  46: "Banana",
                  47: "Apple",
                  48: "Sandwich",
                  49: "Orange",
                  50: "Broccoli",
                  51: "Carrot",
                  52: "Hot dog",
                  53: "Pizza",
                  54: "Donut",
                  55: "Cake",
                  56: "Chair",
                  57: "Couch",
                  58: "Potted plant",
                  59: "Bed",
                  60: "Dining table",
                  61: "Toilet",
                  62: "TV",
                  63: "Laptop",
                  64: "Mouse",
                  65: "Remote",
                  66: "Keyboard",
                  67: "Cell phone",
                  68: "Microwave",
                  69: "Oven",
                  70: "Toaster",
                  71: "Sink",
                  72: "Refrigerator",
                  73: "Book",
                  74: "Clock",
                  75: "Vase",
                  76: "Scissors",
                  77: "Teddy bear",
                  78: "Hair drier",
                  79: "Toothbrush",
                  }

    yolo_results = {}
    index = 1

    for box in boxes:
        yolo_results[index] = {'bbox': [round(float(box[0]), 2), round(float(box[1]), 2), round(float(box[2]), 2),
                                        round(float(box[3]), 2)]}

        image = Image.open(path)

        crop_area = (int(round(float(box[0]), 2)), int(round(float(box[1]), 2)), int(round(float(box[2]), 2)),
                                        int(round(float(box[3]), 2)))

        cropped_image = image.crop(crop_area)

        cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\cropped_imgs\\" + "cropped_im" + str(index) + ".png"

        cropped_image.save(cropped_image_path)

        index = index + 1

    # Open the image file
    image = Image.open(path)  # Replace 'your_image.jpg' with the path to your image file

    # Get the dimensions (size) of the image
    width, height = image.size

    yolo_results[0] = (0, 0, int(round(width, 2)),
                                        int(round(height, 2)))

    image = Image.open(path)

    crop_area = (0, 0, int(round(width, 2)),
                 int(round(height, 2)))

    cropped_image = image.crop(crop_area)

    cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\cropped_imgs\\" + "cropped_im" + str(
        0) + ".png"

    cropped_image.save(cropped_image_path)

    index = 1

    task_with_bbox = []

    # here we will be adding the large image as a task
    task_with_bbox.append((crop_area, cropped_image_path, queue, "scene"))
    count_names = {}
    for c in classes:
        if idToObject[c] in count_names:
            yolo_results[index]["name"] = idToObject[c] + str(count_names[idToObject[c]] + 1)
            cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\cropped_imgs\\" + "cropped_im" + str(
                index) + ".png"
            task_with_bbox.append((yolo_results[index]["bbox"], cropped_image_path, queue, idToObject[c] + str(count_names[idToObject[c]] + 1)))
            count_names[idToObject[c]] = count_names[idToObject[c]] + 1
        else:
            yolo_results[index]["name"] = idToObject[c] + str(1)
            cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\cropped_imgs\\" + "cropped_im" + str(
                index) + ".png"
            task_with_bbox.append((yolo_results[index]["bbox"], cropped_image_path, queue, idToObject[c] + str(1)))
            count_names[idToObject[c]] = 1
        index = index + 1

    # print(yolo_results)

    sorted_tasks = sorted(task_with_bbox, key=lambda x: x[0][0])

    # this will allow us to get the blip2 captions for each image
    pool.starmap(get_blip2_response, sorted_tasks)

    end_time_first = time.time()

    elapsed_time_first = end_time_first - start_time

    print("Elapsed_time_first: ", elapsed_time_first)

    # here we will gather all of the output
    data_list = []
    while not queue.empty():
        data_list.append(queue.get())

    print("DATA_LIST: ", data_list)

    # now we can build a hierachies with the given data

    json_data = {"results" : data_list}

    currentJsonParser = JsonParser.JsonParser(json_data)

    final_json = currentJsonParser.return_final_json()

    history_data = "Current time: " + str(time.time()) + "\n Data: " + final_json + "\n\n"

    # Open the file in read mode to read its existing content
    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\history.txt", 'r') as file:
        existing_data = file.read()

    # Your new data to append to the beginning of the file
    new_data = history_data

    # Combine the new data with the existing data
    combined_data = new_data + existing_data

    # Open the file in write mode to overwrite its content with the combined data
    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\history.txt", 'w') as file:
        file.write(combined_data)

    # write the json into a file
    data = json.loads(final_json)
    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\output.json", 'w') as json_file:
        json.dump(data, json_file, indent=4)

    # from here work on the combination of all the information and then
    # pass the info into gpt-4 turbo for a response given a query

    prefix = "Given the following json answer, time stamp and previous jsons of previous scenes: \n Json Answer: \n"
    betweenItemAndHistory = " Here are the jsons of previous scenes: \n"
    currentQuestionPrompt = "Answer the current question with the following in mind: \n " \
                            "Give me a short and percise answer, use the given json to infer the best answer possible as quick as possible, do not mention" \
                            " coordinates or bounding boxes, do not give me information that is not in the json " \
                            "or history," \
                            " please answer the following question as if you were " \
                            "talking to a person who is blind or has low vision, and lastly treat the responses as if " \
                            "the image is being seen through their own eyes rather than a camera."

    history_json = None
    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\output.json", 'w') as json_file:
        history_json = file.read()



    prompt = prefix + final_json + " TimeStamp: " + str(time.time()) + ". " + betweenItemAndHistory + history_json + currentQuestionPrompt + " " + query

    openai.api_key = "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"

    # here we will be building the string that we will put into content
    gpt4_results = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    response = gpt4_results.choices[0]["message"]["content"]

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Elapsed_time: ", elapsed_time)

    print(response)


# here we will be going through each of the images in the test_images
if __name__=='__main__':
    # here we will be getting multiple images, and then we will be testing them out
    directory_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\"

    model = YOLO('yolov8n.pt')

    file_list = os.listdir(directory_path)

    with Manager() as manager:
        queue = manager.Queue()
        with Pool(processes=20) as pool:
            for file in file_list:
                print("Current file: ", file)
                query = input("What follow up question do you have? (if no questions enter n): ")

                real_time_test(directory_path + file, pool, queue, query, model)