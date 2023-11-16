import json
import multiprocessing as mp
import JsonCombiner.Python.Hierachy
import JsonCombiner.Python.JsonParser
import JsonCombiner.Python.JsonCombiner
from blip2 import blip2
from ocr import ocr
from oneformer import oneformer
from GRiT import GRiT
from GPT4 import GPT4_prompt
from GPT4 import GPT4
import os
import openai
import cv2
import time
import re

'''''''''''
Description: this method allows a user to get the instance segmentations of a
given image

Note: had issues with queue, thus we did not add the output to Queue
'''
def run_oneformer(queue, image_sum):
    results = oneformer.get_oneformer()
    json_results = json.loads(results)
    with open("oneformer/oneformer.json", 'w') as json_file:
        json.dump(json_results, json_file, indent=4)

    # here we will be calling a method that will allow me to
    # call llava_api
    # print(json_results)

    prompt = GPT4_prompt.build_gpt4_prompt(image_sum)

    gpt4_results = GPT4.get_gpt4(prompt)

    queue.put(["gpt4", json.loads(gpt4_results)])

'''''''''''
Description: This method is for llava_api summarizatation
'''
'''''''''
def run_LLaVA_sum(queue, prompt):
    result = llava.get_llava(prompt)
    queue.put(["llava", result])
'''


'''''''''
Description: this method allows users to get text 
information from a given image
'''
def run_easyocr(queue):
    result = ocr.get_easryocr()
    queue.put(["ocr", result])


'''''''''''
Description: this method allows users to get an image summarization of 
an image file
'''
def run_blip2(queue, prompt):
    output = blip2.get_blip2(prompt)
    queue.put(["blip2", output])


'''''''''
Description: this method returns the desnecaptioning of different bounding boxes

NOTE: IN PROGRESS
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

    # we will need to add this into the json
    answer = get_final_json(oneformer_json, json.loads(results_dict["grit"]), results_dict["ocr"], results_dict["gpt4"], start_time)

    # return answer


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

    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark.txt", 'a') as file:
        file.write("Follow up runtime: " + str(elapsed_time) + "s \n")

    # remove all of the escape characters

    json_result = json.loads(answer)

    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\answer.json", "w", encoding="utf-8") as file:
        json.dump(json_result, file, indent=4)

    query = ""
    while query != "n":
        query = input("What follow up question do you have? (if no questions enter n): ")

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark.txt", 'a') as file:
            file.write("Question asked: " + str(query) + "s \n")

        if query != "n":

            start_time_question = time.time()
            # Call ChatGPT
            f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\JsonCombiner\\textFiles\\history.txt", "a")
            f.write(query + "\n")

            prefix = "Given the following json answer the current question with previous questions (if any) in mind: "
            betweenItemAndHistory = " Here are the previous questions asked:"
            currentQuestionPrompt = "Here is the current question, don't answer previous questions but take in mind the " \
                                    "previous things that were mentioned, don't repeat any coordinates, do not mention" \
                                    " coordinates or bounding boxes, do not give me information that is not in the json " \
                                    "or history, if you do not have information about something from the json or hierachy " \
                                    "state that you do not have information about it, also if you are given information about an object's action, choose descriptions answer over" \
                                    " descriptions2, and answer the current question as if " \
                                    "you were talking to a five year old: "

            f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\JsonCombiner\\textFiles\\history.txt", "r")
            history = f.read()

            prompt = prefix + answer + betweenItemAndHistory + history + currentQuestionPrompt + query

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

            stop_time_question= time.time()

            elapsed_time_question = stop_time_question - start_time_question

            with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark.txt", 'a') as file:
                file.write("Answer: " + str(response) + "s \n")

            with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark.txt", 'a') as file:
                file.write("Answer runtime: " + str(elapsed_time_question) + "s \n")

            print("Response: ", response)
    f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\JsonCombiner\\textFiles\\history.txt", "w")
    f.write("")


''''''''''
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

    # now we will pass in the two results into gpt-4
    openai.api_key = "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"

    return results_dict["blip2"]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    '''''''''
    later we will be replacing this code with code that should be able to 
    recieve input from the hololense and return back output, and will determine 
    what to do:
    - either give an image summarization 
    - give an answer to a follow up question 
    - get an answer from google (function calling) 
    '''

    # this is so that we can use ocr
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

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark.txt", 'a') as file:
            file.write("IMAGE NUMBER: " + str(index) + ", Filename: " + filename + " \n")

        image = cv2.imread(directory_path + "\\" + filename)
        cv2.imwrite("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\current.png", image)

        # now we can get the image summarization, and then after
        print("What's in front of me?")

        start_time = time.time()
        image_sum = get_summarization()
        end_time = time.time()

        elapsed_time = end_time - start_time

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark.txt", 'a') as file:
            file.write("Image sum: " + str(image_sum) + "s \n")

        with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\bench_mark.txt", 'a') as file:
            file.write("Image sum runtime: " + str(elapsed_time) + "s \n")

        print("Response: ", image_sum)

        get_followup(image_sum)

        index = index + 1


























    # this code if for webcam intake
    '''''''''''
    # this is where we will be capturing webcam footage which we will then save
    # into an image
    prev_time = 0
    capture_interval = 5

    # this is so that we can get webcam footage
    cap = cv2.VideoCapture(0)

    index = 0
    while True:
        ret, frame = cap.read()

        # this is the case if we did not get something
        if not ret:
            break

        current_time = time.time()

        if current_time - prev_time > capture_interval:
            # print("I went into here!")
            # get_summarization()
            cv2.imwrite("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\current.png", frame)
            cv2.imwrite("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\im" + str(index) + ".png", frame)
            result = get_followup()
            prev_time = current_time
            index = index + 1


        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    '''