import json

import blip2
import requests
from PIL import Image
import multiprocessing as mp

from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

import LLaVA.llava.serve.cli
import torch
import JsonCombiner.Python.Hierachy
import JsonCombiner.Python.JsonParser
import JsonCombiner.Python.main

import test_oneformer


'''
Description: this method allows a user to recieve information from RTMDet
'''
def run_oneformer():
    print("I went into oneformer")
    results = test_oneformer.get_oneformer()
    print(results)

    json_results = json.loads(results)

    with open("oneformer.json", 'w') as json_file:
        json.dump(json_results, json_file, indent=4)

    print("Completed call to oneformer!")

def run_rtmdet(queue):
    global results_rtmdet
    results_rtmdet = "rtm det ran"
    queue.put(["rtmdet", results_rtmdet])

# LLaVA works now!
def run_LLaVA(queue):
    # here we will be calling cli's get_llava method
    global results_LLaVA
    results_LLaVA = "LLaVA ran"
    # llava.llava.serve.cli.get_LLaVA(prompt, img)
    queue.put(["llava", results_LLaVA])

def run_GRiT(queue):
    global results_GRiT
    results_GRiT = "GRiT ran"
    queue.put(["grit", results_GRiT])

def run_blip2(queue, image, processor, model, device):
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    queue.put(["blip2", generated_text])

def run_ocr(queue):
    global results_ocr
    results_ocr = "ocr ran"
    queue.put(["ocr", results_ocr])



def get_followup(img):

    # this is to store the output of the functions that are
    # being run in parallel
    # queue = mp.Queue()

    print("Started follow up")
    # here we will be doing everything in parallel
    # when you want to include parameters, have args=()
    process1 = mp.Process(target=run_oneformer)
    # process2 = mp.Process(target=run_rtmdet, args=(queue,))
    # process3 = mp.Process(target=run_LLaVA, args=(queue,))
    # process4 = mp.Process(target=run_GRiT, args=(queue,))
    # process5 = mp.Process(target=run_ocr, args=(queue,))

    print("Starting process1")
    process1.start()
    # process2.start()
    # process3.start()
    # process4.start()
    # process5.start()

    print("Starting join!")
    try:
        process1.join()
    except Exception as e:
        print(f"An exception occurred when joining process1: {e}")
    print("Joined process1")

    # here we will need to go into the Queue, and we will
    # put the results into a dictionary which we will then print
    oneformer_json = None
    with open("oneformer.json", "r") as json_file:
        oneformer_json = json.load(json_file)

    # here we will be reading from the llava_densecap
    with open('C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\JsonCombiner\\LLaVA_Output\\llava_densecap.json', 'r') as json_file:
        data = json.load(json_file)
    print("loaded_json")

    # here, we will be printing out the results
    # oneformer, rtmdet, llava, grit, ocr
    print("oneformer: ", oneformer_json)
    # print("rtmdet: ", results_dict["rtmdet"])
    print("llava: ", data)
    # print("grit: ", results_dict["grit"])
    # print("ocr: ", results_dict["ocr"])

    answer = get_final_json(oneformer_json, data)

    return ""


def get_final_json(oneformer, llava):
    print("called get_final_json")

    # here we will be calling the main method of the
    # JsonParsers method
    answer = JsonCombiner.Python.main.combine_json(oneformer, llava, "what are the people in front of me wearing?")

    return answer




''''''''''
This is where we will get 
'''
def get_summarization(prompt, img, processor, model, device):

    queue = mp.Queue()

    # we will need to call LLaVA and pass in the prompt and img
    # we will also need to call BLIP2, but we only need to pass in an image
    run_blip2(queue, img, processor, model, device)

    print(queue.get())

    # we need to figure out the best way to combine the two



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("Starting main!")

    # this is need for the initalization of the blip2 model
    # processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16,
    #                                                      resume_download=True)
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("This is the device: ", device)
    #model.to(device)

    print("BLIP2 pretrained complete!")

    file_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\im2.png"
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

    # get_summarization("what is in the given image?", image, processor, model, device)
    get_followup(image)





















































# Potential code structure for server-client code between hololense
'''''''''
# Here in main, we will have a serve like main method that will be constantly running in a for loop
while True:

    # this will be a placeholder till we have the code that allows us to
    # recieve the image from the hololense
    recieved_image = True

    # if we recieved the image then we will start processing
    # the image, we most likely need to find some way to know
    # if it is the first time or if it is the follow up time
    # we will need to have a count down:
    # if the count down is over: we will reset the followup boolean
    # we will also clear out the json which we will pass to gpt4

    # figure out how we can constantly wait for images to come from
    # the hololense and into the program, and when something is recieved
    # we will invoke the methods and will invoke different methods depending on
    # whether it is a follow up question or not

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
        follow_up = False

        if follow_up:
            response = get_followup(image)

            # this is where we will have the code that will allow
            # us to send the information to the hololense
        else:
            response = get_summarization("what is in front of me?", image)

            # this is where we will have the code that will allow
            # us to send the information to the hololense
'''
