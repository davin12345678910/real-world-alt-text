import json
import multiprocessing as mp
import JsonCombiner.Python.Hierachy
import JsonCombiner.Python.JsonParser
import JsonCombiner.Python.JsonCombiner
from miscellaneous import GPT4_prompt
from easy_ocr import easy_ocr
from oneformer import oneformer
from GRiT import GRiT
import os
import openai
import cv2
import time
import GPT4Frontload
from miscellaneous import blip2_v1

'''''''''
Description: this method allows a user to get the instance segmentations of a
given image
Note: had issues with queue, thus we did not add the output to Queue
'''
def run_oneformer(queue, image_sum):
    results = oneformer.get_oneformer()
    json_results = json.loads(results)
    with open("oneformer/oneformer.json", 'w') as json_file:
        json.dump(json_results, json_file, indent=4)

    # here we will be calling gpt4 and building
    # a prompt that we will be passing into gpt4
    prompt = GPT4_prompt.build_gpt4_prompt(image_sum)

    gpt4_results = GPT4.get_gpt4(prompt)

    queue.put(["gpt4", json.loads(gpt4_results)])

'''''''''
Description: this method allows users to get text 
information from a given image
'''
def run_easyocr(queue):
    result = easy_ocr.get_easryocr()
    queue.put(["easy_ocr", result])

'''''''''
Description: this method allows users to get an image summarization of 
an image file
'''
def run_blip2(queue, prompt):
    output = blip2_v1.get_blip2(prompt)
    queue.put(["blip2_endpoint_call", output])

'''''''''
Description: gives a user the dense captions from a model called GRiT
'''
def run_GRiT(queue):
    results = GRiT.get_grit()
    queue.put(["grit", results])

'''''''''
Description: this method will allow a user to ask a follow 
up question to an image summarization that is given
'''
def get_followup(image_sum):

    # this is to store the output of the functions that are
    # being run in parallel
    queue = mp.Queue()

    # here we will be doing everything in parallel
    # when you want to include parameters, have args=()
    start_time = time.time()
    process1 = mp.Process(target=run_oneformer, args=(queue, image_sum,))
    process2 = mp.Process(target=run_easyocr, args=(queue,))
    process3 = mp.Process(target=run_GRiT, args=(queue,))

    # here we will start all of the processes
    process1.start()
    process2.start()
    process3.start()

    # here we will be joining all of the processes
    try:
        process1.join()
        process2.join()
        process3.join()
    except Exception as e:
        print(f"An exception occurred when joining process1: {e}")

    # here we will need to go into the Queue, and we will
    # put the results into a dictionary which we will then print
    oneformer_json = None
    with open("oneformer/oneformer.json", "r") as json_file:
        oneformer_json = json.load(json_file)

    # this is where we will store all of the results that we got from the
    # different processes
    results_dict = {}
    while (not queue.empty()):
        current = queue.get()
        results_dict[current[0]] = current[1]

    # here we will call a method that will allow us to build the json to pass into gpt4 and to
    # start conversations with gpt4
    get_final_json(oneformer_json, json.loads(results_dict["grit"]), results_dict["easy_ocr"],
                            results_dict["gpt4"], start_time)

'''''''''
Description: This method calls JsonCombiner which will combine all of the 
various json information that we have recieved
'''
def get_final_json(oneformer, llava, ocr, gpt4, start_time):

    # here we will be calling the main method of the
    # JsonParsers method
    answer = JsonCombiner.Python.JsonCombiner.combine_json(oneformer, llava, ocr, gpt4)

    end_time = time.time()

    elapsed_time = end_time - start_time

    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_main.txt", 'a') as file:
        file.write("Follow up runtime: " + str(elapsed_time) + "s \n")

    json_result = json.loads(answer)

    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\answer.json", "w", encoding="utf-8") as file:
        json.dump(json_result, file, indent=4)

    # here we will be starting our queries with gpt4
    query = ""
    while query != "n":
        query = input("What follow up question do you have? (if no questions enter n): ")

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_main.txt", 'a') as file:
            file.write("Question asked: " + str(query) + "s \n")

        # here we are going to be checking if we will be doing any more queries or not
        if query != "n":
            start_time_question = time.time()
            # Call ChatGPT
            f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\JsonCombiner\\textFiles\\history.txt",
                     "a")
            f.write(query + "\n")

            prefix = "Given the following json answer and previous questions (if any): \n Json Answer: \n"
            betweenItemAndHistory = " Here are the previous questions asked: \n"
            currentQuestionPrompt = "Answer the current question with the following in mind: don't answer previous questions but take in mind the " \
                                    "previous things that were mentioned, don't repeat any coordinates, do not mention" \
                                    " coordinates or bounding boxes, do not give me information that is not in the json " \
                                    "or history, if you do not have information about something from the json or hierachy " \
                                    "state that you do not have information about it, " \
                                    " please answer the following question as if you were " \
                                    "talking to a person who is blind or has low vision, and lastly treat the responses as if " \
                                    "the image is being seen through their own eyes rather than a camera."

            f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\JsonCombiner\\textFiles\\history.txt",
                     "r")
            history = f.read()

            prompt = prefix + answer + "\n" + betweenItemAndHistory + history + "\n" + currentQuestionPrompt + query

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

            stop_time_question = time.time()

            elapsed_time_question = stop_time_question - start_time_question

            with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_main.txt", 'a') as file:
                file.write("Answer: " + str(response) + "s \n")

            with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_main.txt", 'a') as file:
                file.write("Answer runtime: " + str(elapsed_time_question) + "s \n")

            print("Response: ", response)
    f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\JsonCombiner\\textFiles\\history.txt", "w")
    f.write("")

'''''''''
Description: This is where we will get image summarization information 
'''
def get_summarization():

    queue = mp.Queue()

    # we will need to call LLaVA and pass in the prompt and img
    # we will also need to call BLIP2, but we only need to pass in an image
    prompt_blip2 = "what is in the given image?"
    process1 = mp.Process(target=run_blip2, args=(queue, prompt_blip2,))

    process1.start()

    process1.join()

    results_dict = {}
    while (not queue.empty()):
        current = queue.get()
        results_dict[current[0]] = current[1]

    return results_dict["blip2_endpoint_call"]


def get_followup_gpt4frontload(path):
    json_result = GPT4Frontload.GPT_front_load.get_gpt4_frontload(path)
    followup_querying(json_result)


# here we will be having the querying portion of the task
def followup_querying(json_result):

    # here we will be starting our queries with gpt4
    query = ""
    while query != "n":
        query = input("What follow up question do you have? (if no questions enter n): ")

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_main.txt", 'a') as file:
            file.write("Question asked: " + str(query) + "s \n")

        # here we are going to be checking if we will be doing any more queries or not
        if query != "n":
            start_time_question = time.time()
            # Call ChatGPT
            f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\JsonCombiner\\textFiles\\history.txt",
                     "a")
            f.write(query + "\n")

            prefix = "Given the following json answer and previous questions (if any): \n Json Answer: \n"
            betweenItemAndHistory = " Here are the previous questions asked: \n"
            currentQuestionPrompt = "Answer the current question with the following in mind: don't answer previous questions but take in mind the " \
                                    "previous things that were mentioned, don't repeat any coordinates, do not mention" \
                                    " coordinates or bounding boxes, do not give me information that is not in the json " \
                                    "or history, if you do not have information about something from the json or hierachy " \
                                    "state that you do not have information about it, " \
                                    " please answer the following question as if you were " \
                                    "talking to a person who is blind or has low vision, and lastly treat the responses as if " \
                                    "the image is being seen through their own eyes rather than a camera."

            f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\JsonCombiner\\textFiles\\history.txt",
                     "r")
            history = f.read()

            prompt = prefix + json_result + "\n" + betweenItemAndHistory + history + "\n" + currentQuestionPrompt + query

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

            stop_time_question = time.time()

            elapsed_time_question = stop_time_question - start_time_question

            print("Response: ", response)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    '''''''''''
    later we will be replacing this code with code that should be able to 
    recieve input from the hololense and return back output, and will determine 
    what to do:
    - either give an image summarization 
    - give an answer to a follow up question 
    - get an answer from google (function calling) 
    '''

    # this is so that we can use easy_ocr
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["REPLICATE_API_TOKEN"] = "r8_HfRAhxwo6UJWnpdEPkLhpwC9HIRxGzn0fHb38"

    # here you will need to write the current image into the /test-image/current.jpg
    # we will go into a for loop, and we will take a snap shot and then run
    # get_followup to see how long it takes

    # here we will be getting multiple images, and then we will be testing them out
    directory_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image"

    file_list = os.listdir(directory_path)

    index = 1

    # for testing with various test images
    for filename in file_list:
        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_main.txt", 'a') as file:
            file.write("IMAGE NUMBER: " + str(index) + ", Filename: " + filename + " \n")

        image = cv2.imread(directory_path + "\\" + filename)
        cv2.imwrite("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\current.png", image)

        # now we can get the image summarization, and then after
        print("What's in front of me?")

        start_time = time.time()
        image_sum = get_summarization()
        end_time = time.time()

        elapsed_time = end_time - start_time

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_main.txt", 'a') as file:
            file.write("Image sum: " + str(image_sum) + "s \n")

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_main.txt", 'a') as file:
            file.write("Image sum runtime: " + str(elapsed_time) + "s \n")

        print("Response: ", image_sum)

        get_followup(image_sum)

        index = index + 1

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark_main.txt", 'a') as file:
            file.write("\n")
