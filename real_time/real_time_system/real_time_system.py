from ultralytics import YOLO

# from transformers import AutoProcessor, Blip2Processor, Blip2ForConditionalGeneration
# import requests
from PIL import Image
import os
import blip2_endpoint
import openai
import time
# import replicate
from multiprocessing import Pool, Manager
from JsonParser import JsonParser

import json

'''''''''
Definition:
this allows a user to get a response from blip2 and store the response in the multiprocessing queue

Parameters:
bbox - this is the bounding box of the current object in the image 
path - this is the path of the current cropped image 
queue - this is where we will be storing the output
name - this is the name of the object that the cropped image is focusing on 

Returns:
None
'''
def get_blip2_response(bbox, path, queue, name):

    response = blip2_endpoint.get_blip2(path, name)
    bbox_formatted = [[bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]]]
    queue.put({"name" : name, "bbox": bbox_formatted, "description" : response})


'''''''''
Definition:
This method will allow a person to ask questions about a given image 

Parmaters: 
path - this is the path of the current image 
pool - this is the processing pool that tasks will be using in order to run parallel code 
queue - this is where we will be storing the results of tasks 
query - this is the question that we will be asking about the given image 
model - this is the yolo model that we will be using for object detection 

Returns:
None
'''
def real_time_test(path, pool, queue, query, model):

    start_time = time.time()

    # this is the results from the yolo mode
    # this will give us the objects that yolo can detect in the given image
    results = model(
        [path])

    end_time3 = time.time()

    print("Elapsed_time yolo: ", (end_time3 - start_time))

    '''''''''
    Here we will be processing all of the yolo information which we will 
    later use in order to get the bounding boxes for each object which will
    be used to crop the image 
    '''
    boxes = None
    classes = None
    for key in results:
        boxes = key.boxes.xyxy.tolist()
        classes = key.boxes.cls.tolist()

    # these are the different objects that yolo cna detect
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

    # here we will be getting the cropped images as well as all of the objects that were detected
    yolo_results = {}
    index = 1

    for box in boxes:
        yolo_results[index] = {'bbox': [round(float(box[0]), 2), round(float(box[1]), 2), round(float(box[2]), 2),
                                        round(float(box[3]), 2)]}
        image = Image.open(path)
        crop_area = (int(round(float(box[0]), 2)), int(round(float(box[1]), 2)), int(round(float(box[2]), 2)),
                                        int(round(float(box[3]), 2)))
        cropped_image = image.crop(crop_area)
        cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real_time\\cropped_imgs\\" + "cropped_im" + str(index) + ".png"
        cropped_image.save(cropped_image_path)

        index = index + 1

    # here we will be saving the cropped image of the whole image itself
    image = Image.open(path)
    width, height = image.size
    yolo_results[0] = (0, 0, int(round(width, 2)),
                                        int(round(height, 2)))
    image = Image.open(path)
    crop_area = (0, 0, int(round(width, 2)),
                 int(round(height, 2)))
    cropped_image = image.crop(crop_area)
    cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real_time\\cropped_imgs\\" + "cropped_im" + str(
        0) + ".png"
    cropped_image.save(cropped_image_path)


    # here we will be making all of the tasks which we will be passing into blip2
    index = 1
    task_with_bbox = []

    # here we will be adding the large image as a task
    task_with_bbox.append((crop_area, cropped_image_path, queue, "scene"))
    count_names = {}
    for c in classes:
        if idToObject[c] in count_names:
            yolo_results[index]["name"] = idToObject[c] + str(count_names[idToObject[c]] + 1)
            cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real_time\\cropped_imgs\\" + "cropped_im" + str(
                index) + ".png"
            task_with_bbox.append((yolo_results[index]["bbox"], cropped_image_path, queue, idToObject[c] + str(count_names[idToObject[c]] + 1)))
            count_names[idToObject[c]] = count_names[idToObject[c]] + 1
        else:
            yolo_results[index]["name"] = idToObject[c] + str(1)
            cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real_time\\cropped_imgs\\" + "cropped_im" + str(
                index) + ".png"
            task_with_bbox.append((yolo_results[index]["bbox"], cropped_image_path, queue, idToObject[c] + str(1)))
            count_names[idToObject[c]] = 1
        index = index + 1

    # here we will be sorting all of the imanges from left to right
    sorted_tasks = sorted(task_with_bbox, key=lambda x: x[0][0])

    # run all of the blip2 tasks in parallel
    pool.starmap(get_blip2_response, sorted_tasks)

    end_time_first = time.time()

    elapsed_time_first = end_time_first - start_time

    print("Elapsed_time_first: ", elapsed_time_first)

    # here we will gather all of the output
    data_list = []
    while not queue.empty():
        data_list.append(queue.get())

    print("DATA_LIST: ", data_list)

    # this is where we build the resulting json with all of the information that we need
    json_data = {"results" : data_list}

    currentJsonParser = JsonParser(json_data)

    final_json = currentJsonParser.return_final_json()


    # this is where we will be storing previous data
    '''''''''
    history_data = "Current time: " + str(time.time()) + "\n Data: " + final_json + "\n\n"

    # Open the file in read mode to read its existing content
    with open("/real_time\\history.txt", 'r') as file:
        existing_data = file.read()

    # Your new data to append to the beginning of the file
    new_data = history_data

    # Combine the new data with the existing data
    combined_data = new_data + existing_data

    # Open the file in write mode to overwrite its content with the combined data
    with open("/real_time\\history.txt", 'w') as file:
        file.write(combined_data)
    '''

    # write the json into a file
    data = json.loads(final_json)
    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real_time\\output.json", 'w') as json_file:
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

    # history_json = None
    # with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real_time\\output.json", 'w') as json_file:
    #    history_json = file.read()
    # prompt = (prefix + final_json + " TimeStamp: " + str(time.time()) + ". " + betweenItemAndHistory + history_json +
    #          currentQuestionPrompt + " " + query)

    prompt = (prefix + final_json + " TimeStamp: " + str(time.time()) + ". " +
              currentQuestionPrompt + " " + query)

    openai.api_key = "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"

    # here we will be building the string that we will put into content
    gpt4_results = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # this is the response given the json we constructed and the current question
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