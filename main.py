import json
import multiprocessing as mp
import JsonCombiner.Python.Hierachy
import JsonCombiner.Python.JsonParser
import JsonCombiner.Python.main
from blip2 import blip2
from llava import llava
from llava import llava_prompt
from ocr import ocr
from oneformer import oneformer
import os
import openai
import cv2

'''''''''''
Description: this method allows a user to get the instance segmentations of a
given image

Note: had issues with queue, thus we did not add the output to Queue
'''
def run_oneformer(queue, image_sum):
    # print("I went into oneformer")
    results = oneformer.get_oneformer()
    json_results = json.loads(results)
    with open("oneformer/oneformer.json", 'w') as json_file:
        json.dump(json_results, json_file, indent=4)

    # here we need to call oneformer prompt
    prompt = llava_prompt.build_llava_prompt(results, image_sum)

    # here we will be calling llava.py
    result = llava.get_llava(prompt)

    # print("This is LLaVA result: ", result)

    # turn the response into a json
    result_json = json.loads(result)

    # print("llava: ", result)

    queue.put(["llava", result_json])
    # print("I finished calling oneformer")

'''''''''''
Description: This method is for llava summarizatation
'''
def run_LLaVA_sum(queue, prompt):
    result = llava.get_llava(prompt)
    queue.put(["llava", result])


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
def run_blip2(queue,):
    output = blip2.get_blip2()
    queue.put(["blip2", output])


'''''''''
Description: this method returns the desnecaptioning of different bounding boxes

NOTE: IN PROGRESS
'''
def run_GRiT(queue):
    global results_GRiT
    results_GRiT = "GRiT ran"
    queue.put(["grit", results_GRiT])



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
    process1 = mp.Process(target=run_oneformer, args=(queue, image_sum,))
    process2 = mp.Process(target=run_easyocr, args=(queue,))

    # here we will start all of the processes
    process1.start()
    process2.start()

    # print("I am going to join all of these processes")
    # here we will be joining all of the processes
    try:
        process1.join()
        process2.join()
        # print("I joined all of these processes")
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

    answer = get_final_json(oneformer_json, results_dict["llava"], results_dict["ocr"])

    '''''''''
    with open('C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\output.txt', 'a') as file:
        file.write('\n' + answer)
    '''

    # return answer


'''''''''
Description: This method calls JsonCombiner which will combine all of the 
various json information that we have recieved
'''
def get_final_json(oneformer, llava, ocr):

    # here we will be calling the main method of the
    # JsonParsers method
    answer = JsonCombiner.Python.main.combine_json(oneformer, llava, ocr)

    # print("Answer: ", answer)

    query = ""
    while query != "n":
        query = input("What follow up question do you have? (if no questions enter n): ")

        if query != "n":
            # Call ChatGPT
            f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\JsonCombiner\\textFiles\\history.txt", "a")
            f.write(query + "\n")

            prefix = "Given the following json answer the current question with previous questions (if any) in mind: "
            betweenItemAndHistory = " Here are the previous questions asked:"
            currentQuestionPrompt = "Here is the current question, don't answer previous questions but take in mind the previous things that were mentioned, don't repeat any coordinates, do not mention coordinates or bounding boxes, do not give me information that is not in the json or history, if you do not have information about something from the json or hierachy state that you do not have information about it, and answer the current question as if you were talking to a five year old: "

            f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\JsonCombiner\\textFiles\\history.txt", "r")
            history = f.read()

            prompt = prefix + answer + betweenItemAndHistory + history + currentQuestionPrompt + query

            openai.api_key = "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"

            # here we will be building the string that we will put into content
            gpt4_results = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            response = gpt4_results.choices[0]["message"]["content"]
            print("Response: ", response)


''''''''''
Description: This is where we will get image summarization information 
'''
def get_summarization():

    queue = mp.Queue()

    # we will need to call LLaVA and pass in the prompt and img
    # we will also need to call BLIP2, but we only need to pass in an image
    process1 = mp.Process(target=run_blip2, args=(queue,))
    prompt = "what is in front of me"
    process2 = mp.Process(target=run_LLaVA_sum, args=(queue, prompt,))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    results_dict = {}
    while (not queue.empty()):
        current = queue.get()
        results_dict[current[0]] = current[1]

    # now we will pass in the two results into gpt-4
    openai.api_key = "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"

    # here we will be building the string that we will put into content
    prompt_sum = "Given these two image captions, give me a combined version of these two: (" + results_dict["blip2"] + "), (" + results_dict["llava"] + ")"

    gpt4_results = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_sum}
        ]
    )

    # print("Follow up: ", gpt4_results.choices[0]["message"]["content"])
    # print("")

    return gpt4_results.choices[0]["message"]["content"]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # this is so that we can use ocr
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["REPLICATE_API_TOKEN"] = "r8_HfRAhxwo6UJWnpdEPkLhpwC9HIRxGzn0fHb38"

    # here you will need to write the current image into the /test-image/current.jpg
    # we will go into a for loop, and we will take a snap shot and then run
    # get_followup to see how long it takes

    # here we will be getting multiple images, and then we will be testing them out
    directory_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image"

    file_list = os.listdir(directory_path)

    for filename in file_list:

        image = cv2.imread(directory_path + "\\" + filename)
        cv2.imwrite("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\current.png", image)

        # now we can get the image summarization, and then after
        print("What's in front of me?")

        image_sum = get_summarization()

        print("Response: ", image_sum)

        get_followup(image_sum)




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