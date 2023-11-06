import subprocess
import json
from shapely.geometry import Polygon
import pycocotools.mask  as _mask
from imantics import Polygons, Mask
import numpy as np
import requests

'''''''''
This method will allow us to pass in a string 
with bounding box information of an object
'''


def build_LLaVA_Query():

    # here we will make a call to the oneformer endpoint
    endpoint_url = "http://localhost:5000/get_oneformer_results"

    image_file_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\im2.png"

    image = {'image' : open(image_file_path, 'rb')}

    results = None
    # here we will make a request to the endpoint
    try:
        response = requests.post(endpoint_url, files=image)
        if response.status_code == 200:
            # print('Response from server:', response.text)
            results = response.text
        else:
            print('Error:', response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print('Error:', e)


    # now we will sort all of the objects by their x coordinates
    sort_horizontal = {}

    # now we just need to get the x values, and then we will
    # put them in the sort_horizontal
    for object in json.loads(results)["results"]:
        coordinates = object["bbox"]
        print(coordinates)
        x_coord = (float(coordinates[0][0]) + float(coordinates[2][0])) / 2
        sort_horizontal[x_coord] = object

    sorted_keys = sorted(sort_horizontal.keys())


    llava_objects = []

    for key in sorted_keys:
        llava_objects.append(sort_horizontal[key])

    '''''''''''''''
    A computer vision model found the following objects in this image, 
    listed in order from top left to bottom right: 
    {"person1", polygon}, {"person2", polygon}, {"bench1", polygon}, 
    {"dog1", polygon}, {"dog2", polygon}, {"dog3", polygon}, and {"dog4", polygon}. 
    Your goal is to write a dense caption of each object for someone who is blind. 
    For each object given, if you cannot find any given information in the given bbox of
    an object do not include any information. Your description of each object 
    should only include information you are fairly confident about. 
    When outputting your response, follow this json structure:
    [ {"name": "an object from the list of objects", "caption": "a dense caption of the object"}]
    '''''
    llava_string = ""

    llava_string += "A computer vision model found the following objects in " \
                    "this image, listed in order from left to right: "

    index = 0
    for current in llava_objects:
        if index == 0:
            llava_string += str({"name" : current["name"], "bbox" : current["bbox"]})
        else:
            llava_string += ", " + str({"name" : current["name"], "bbox" : current["bbox"]})

        index = index + 1

    llava_string += ". Your goal is to write a dense caption of each object" \
                    " for someone who is blind." \
                    " For each object given, if you cannot find any given information about an object given the bbox of" \
                    " an object do not include any information. " \
                    "Also, never include any information about gender, such as man or woman. " \
                    "If you are not sure, do not include the information." \
                    " When outputting your response, follow this json structure:{results : [{\"name\":" \
                    " \"an object from the list of objects\", \"caption\": \"a dense caption of the object\"}]}"


    # now we will need to pass this into llava
    return llava_string



if __name__ == '__main__':
    result = build_LLaVA_Query()
    print(result)


























# this code is for RTMDet
'''''''''''''''
# this is where we will be getting id to object information
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

# this is where we will be running RTMDet on a given input image
subprocess.run(
    "cd mmdetection && conda deactivate && conda activate kitchen_access && python demo/image_demo.py TestImage/im2.png configs/rtmdet/rtmdet-ins_x_8xb16-300e_coco.py --weights checkpoints/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth --pred-score-thr 0.5",
    shell=True)

# here we will retrive the information that RTMDet got and parse throught it
with open('mmdetection/outputs/preds/im2.json', 'r') as file:
    file_contents = file.read()

# now we will turn the information into a json
instance_segmentation_json = json.loads(file_contents)

# this is where we will be storing the current objects that we have looked over
current_objects = {}

# this is where we will get the number of each item
object_to_count = {}

# this is where we will store the id of the current object we are on, which will act as the
# key for each itme in the map
id = 0

# here we will look through each of the objects in the json
for object in instance_segmentation_json.get("labels"):

    # if we have a object of the same name, we will add a number to the back of the
    # current oject name
    if object_to_count.__contains__(idToObject[object]):
        object_to_count[idToObject[object]] = object_to_count[idToObject[object]] + 1
        current_objects[id] = {"name": idToObject[object] + str(object_to_count[idToObject[object]])}

    # this is if this is the first time a object of the given name shows up
    else:
        current_objects[id] = {"name": idToObject[object] + str(1)}
        object_to_count[idToObject[object]] = 1

    id = id + 1

# here we will be adding the confidence of each of the objects
id = 0
for score in instance_segmentation_json.get("scores"):
    # here we will be adding in the scores to each of the objects
    current_objects[id]["score"] = score

    id = id + 1

# then we will be putting in the polygon into the json for each of the objects
id = 0

centers = {}
for bbox in instance_segmentation_json.get("bboxes"):

    # here we will be filtering out the things that meet the threshold of
    if current_objects[id]["score"] >= 0.3:

        # here we will be adding in the scores to each of the objects
        current_objects[id]["bbox"] = bbox

        # here we will take the first two values as x and y
        # then the next two will be halved and be added to x and y
        x = float(bbox[0])
        y = float(bbox[1])
        dx = float(bbox[2])
        dy = float(bbox[3])

        center = (x + dx)

        # there might be a possibility of multiple
        # objects with the same x value
        if center not in centers:
            centers[center] = []
        centers[center].append(id)

    else:
        # print("I went into here!")
        del current_objects[id]
    id = id + 1

sorted_centers = sorted(centers.items())

# then we will be making a list of all of the object names to their polygon
# and then we will return it
llava_objects = []

# put everything in this format
for x, y in sorted_centers:

    # now we will go through each of the id's for each x
    for object in centers[x]:
        # we will need to round the number in the bbox
        coor1 = round(float(current_objects[object]["bbox"][0]), 1)
        coor2 = round(float(current_objects[object]["bbox"][1]), 1)
        coor3 = round(float(current_objects[object]["bbox"][2]), 1)
        coor4 = round(float(current_objects[object]["bbox"][3]), 1)

        coordinates = [coor1, coor2, coor3, coor4]

        llava_objects.append({"name": current_objects[object]["name"], "bbox": coordinates})
'''