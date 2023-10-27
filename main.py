# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import blip2

import requests
from PIL import Image

import llava.llava.serve.cli


'''
Description: this method will allow us to recieve json information from RTMDet
'''
def run_oneFormer(img):
    print("running oneFormer")

def run_rtmdet(img):
    print("running rtmdet")

# LLaVA works now!
def run_LLaVA(img, prompt):
    # here we will be calling cli's get_llava method
    return llava.llava.serve.cli.get_LLaVA(prompt, img)

def run_GRiT(img):
    print("running GRiT")

def run_blip2(img):
    caption = blip2.getBlip2(img)
    return caption

def get_followup(img):
    # we will need to get the object from RTMDet
    # rtmDet, already takes in its own image
    instance_segmentation = run_rtmdet(img)



''''''''''
This is where we will get 
'''
def get_summarization(prompt, img):

    # this will be for image summarization
    llava = run_LLaVA(prompt, img)

    # here we will be getting the response from blip2
    blip2 = run_blip2(img)

    # we most likely will want to make a json that we
    # will be sending to gpt4, test this out

    return llava + blip2


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Here in main, we will have a serve like main method that will be constantly running in a for loop
    while True:

        # this will be a placeholder till we have the code that allows us to
        # recieve the image from the hololense
        recieved_image = True

        # if we recieved the image then we will start processing
        # the image, we most likely need to find some way to know
        # if it is the first time or if it is the

        # we need some way to know
        if recieved_image:
            # this is where we will be
            # this is where we will be getting the image from the hololense
            # this will depend on your path name
            file_path = "C:\\Users\\davin123\\PycharmProjects\\makeability_real-world-alt-text\\test-image\\king_county_buses.jpg"
            image = None
            # this is where we will be getting the image object which we will
            # be passing into the summarization method
            try:
                image = Image.open(file_path).convert('RGB')
                # Now you can work with the 'image' object
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except Exception as e:
                print(f"An error occurred: {str(e)}")

            # here we will need to determine if this is a follow up question or not

            get_summarization("what is in front of me?", image)

