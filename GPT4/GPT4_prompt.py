import subprocess
import json
from shapely.geometry import Polygon
import pycocotools.mask  as _mask
from imantics import Polygons, Mask
import numpy as np
import requests

import replicate
import os

def build_gpt4_prompt(image_sum):

    # now we will sort all of the objects by their x coordinates
    sort_horizontal = {}

    # now we just need to get the x values, and then we will
    # put them in the sort_horizontal
    with open("oneformer/oneformer.json", 'r') as json_file:
        results = json_file.read()

    # print("RESULTS: ", results)

    for object in json.loads(results)["results"]:
        coordinates = object["bbox"]
        # print(coordinates)
        x_coord = (float(coordinates[0][0])) / 2
        sort_horizontal[x_coord] = object

    sorted_keys = sorted(sort_horizontal.keys())

    # here we will get all of the llava_api objects
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

    # this is where we will be storing the objects, which will only include name and bbox information
    index = 0
    for current in llava_objects:
        if index == 0:
            llava_string += str({"name" : current["name"], "bbox" : current["bbox"]})
        else:
            llava_string += ", " + str({"name" : current["name"], "bbox" : current["bbox"]})

        index = index + 1

    llava_string += "Your goal is to make a json with a dense caption for each object for someone who is blind. " \
                    "Include as much information as you can about color, clothing, action and relationship to " \
                    "other objects in the dense captions. Also, only include information that you are highly " \
                    "confident about. In addition, the image summarization of this image was: " + image_sum + "." \
                    " When " \
                    "outputting your response, make a json with the following json structure" \
                    " and do not give duplicate information: " \
            "{response : [{\"name\":" \
                    " \"an object from the list of objects\", \"caption\": \"a dense caption of the object\"}, ...]}"

    # print("LLaVA_string: ", llava_string)

    return llava_string
