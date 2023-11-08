import subprocess
import os
import json
import pycocotools.mask  as _mask
from imantics import Polygons, Mask
import cv2
import numpy as np


# here we will need to go into the right directory
subprocess.run("cd mmdetection && conda deactivate && conda activate openmmlab && python demo/image_demo.py demo/demo.jpg rtmdet-ins_x_8xb16-300e_coco.py --weights rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth --pred-score-thr 0.5", shell=True)

# now let's get the json results and output the results
with open('C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\mmdetection\\outputs\\preds\\demo.json', 'r') as file:
    file_contents = file.read()

# turn the file_contents into a json
instance_segmentation_json = json.loads(file_contents)

# print(instance_segmentation_json.get("labels"))

idToObject = {0 : "person",
              1 : "Bicycle",
              2 : "car",
              3 : "Motorcycle",
              4 : "Airplace",
              5 : "Bus",
              6 : "Train",
              7 : "Truck",
              8 : "Boat",
              9 : "Traffic Light",
              10 : "Fire Hydrant",
              11 : "Stop sign",
              12 : "Parking Meter",
              13 : "Bench",
              14 : "Bird",
              15 : "Cat",
              16 : "Dog",
              17 : "Horse",
              18 : "Sheep",
              19 : "Cow",
              20 : "Elephant",
              21 : "Bear",
              22 : "Zebra",
              23 : "Giraffe",
              24 : "Backpack",
              25 : "Umbrella",
              26 : "Handbag",
              27 : "Tie",
              28 : "Suitcase",
              29 : "Frisbee",
              30 : "Skis",
              31 : "Snowboard",
              32 : "Sports ball",
              33 : "Kite",
              34 : "Baseball bat",
              35 : "Baseball glove",
              36 : "Skateboard",
              37 : "Surfboard",
              38 : "Tennis racket",
              39 : "Bottle",
              40 : "Wine glass",
              41 : "Cup",
              42 : "Fork",
              43 : "Knife",
              44 : "Spoon",
              45 : "Bowl",
              46 : "Banana",
              47 : "Apple",
              48 : "Sandwich",
              49 : "Orange",
              50 : "Broccoli",
              51 : "Carrot",
              52 : "Hot dog",
              53 : "Pizza",
              54 : "Donut",
              55 : "Cake",
              56 : "Chair",
              57 : "Couch",
              58 : "Potted plant",
              59 : "Bed",
              60 : "Dining table",
              61 : "Toilet",
              62 : "TV",
              63 : "Laptop",
              64 : "Mouse",
              65 : "Remote",
              66 : "Keyboard",
              67 : "Cell phone",
              68 : "Microwave",
              69 : "Oven",
              70 : "Toaster",
              71 : "Sink",
              72 : "Refrigerator",
              73 : "Book",
              74 : "Clock",
              75 : "Vase",
              76 : "Scissors",
              77 : "Teddy bear",
              78 : "Hair drier",
              79 : "Toothbrush",
            }

# print(idToObject)

# here we will be getting the labels, and then we will put them into a map

# print(file_contents)

# now we will be making the json object that will have all of the data where everything is decoded
# and properly structured

# and we will be adding it by their ids, and the ids will start from 0 and go up
id = 0
json_objects = {}
for objects in instance_segmentation_json.get("labels"):
     json_objects[id] = {"name" : idToObject[objects]}
     id = id + 1

# now we will be adding on the additional data such as the condfidence,
# the polygon, the bounding box, and then the

# here we will be adding in the confidence
id = 0
for score in instance_segmentation_json.get("scores"):

    # here we will be adding in the scores to each of the objects
    json_objects[id]["score"] = score

    id = id + 1


# now we will be adding in the bounding box
id = 0
for bbox in instance_segmentation_json.get("bboxes"):

    # here we will be adding in the scores to each of the objects
    json_objects[id]["bbox"] = bbox

    id = id + 1


# now we will be getting the polygon
id = 0
for mask in instance_segmentation_json.get("masks"):

    # here we will be filtering out the things that meet the threshold of
    if json_objects[id]["score"] >= 0.5:
        # here we will be adding in the scores to each of the objects
        json_objects[id]["mask"] = Mask(_mask.decode(mask)).polygons().points

        # print(Mask(_mask.decode(mask)).polygons().points)
    else:
        del json_objects[id]

    id = id + 1


print("These are the json objects: ", json_objects)

# here we will be outputing the result of the image
image = cv2.imread("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\mmdetection\\demo\\demo.jpg")

for id in json_objects:
    # print(len(json_objects[id]["mask"]))

    i = 0
    for array in json_objects[id]["mask"]:
        # print("I went in!")
        poly_coords = np.array(array)
        cv2.polylines(image, [poly_coords], isClosed=True, color=(0, 255, 0), thickness=2)
        if i == 0:
            text_position = tuple(poly_coords[0])
            cv2.putText(image, json_objects[id]["name"], text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

        i = i + 1

# here we will be showing the image with all of the markings
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()






#Below is code to output instance segmentation results:
'''''''''
# here we will be outputing the result of the image
image = cv2.imread("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\mmdetection\\demo\\test#1.jpg")

for id in json_objects:
    # print(len(json_objects[id]["mask"]))

    i = 0
    for array in json_objects[id]["mask"]:
        # print("I went in!")
        poly_coords = np.array(array)
        cv2.polylines(image, [poly_coords], isClosed=True, color=(0,255,0), thickness=2)
        if i == 0:
            text_position = tuple(poly_coords[0])
            cv2.putText(image, json_objects[id]["name"], text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        i = i + 1


#here we will be showing the image with all of the markings
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''




