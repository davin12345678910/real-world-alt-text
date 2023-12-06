import GPT4Frontload.GPT_front_load
from multiprocessing import Pool, Manager
import openai
import cv2
import time
from PIL import Image
import os
import blip2_endpoint_call
import real_time
import real_time.real_time_system
import real_time.real_time_system.blip2_endpoint
import openai
import time
# import replicate
from multiprocessing import Pool, Manager
import real_time
import json

# this will be used for the object detection for YOLO
from ultralytics import YOLO


# code for the real-time mode
import real_time.real_time_system.real_time_system


json_reuse = None

# now we will be including the frontloading code
'''''''''''
Definition
This will be where we can start up the followup system for gpt4 frontload

Parameters:
path - this is the path to the current image 

Returns:
None
'''
def followup_gpt4frontload(path):
    start_time = time.time()
    json_result = GPT4Frontload.GPT_front_load.get_gpt4_frontload(path)
    end_time = time.time()

    with open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\benchmark_finalsys.txt", 'a') as file:
        file.write("Frontloading Runtime: " + str(end_time - start_time) + "\n")

    global json_reuse

    json_reuse = json_result


'''''''''''
Definition:
Gives the user the response to a given query and the current image for our gpt4 frontload system  

Parameters:
query - this is the current question a person is asking about an image 
json_result - this is the json that we constructed to represent the image that a person is wondering about 

Returns:
an answer to a given question anf image of a json 
'''
def get_followup_answer_gpt4frontload(query, json_result):
    f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\textFiles\\history.txt",
             "a")
    f.write(query + "\n")

    # modify the prompt so it doesn't say anything about the json information, just say i don't have enough information
    prefix = "Given the following json answer and previous questions (if any): \n Json Answer: \n"
    betweenItemAndHistory = " Here are the previous questions asked: \n"
    currentQuestionPrompt = "Answer the current question with the following in mind: please be brief and short in your responses \n" \
                            "please don't answer previous questions but take in mind the, try to infer answers if needed " \
                            "previous things that were mentioned, don't repeat any coordinates, do not mention" \
                            " coordinates or bounding boxes, do not give me information that is not in the json " \
                            "or history, if you do not have information about something from the json or hierachy " \
                            "state that you do not have information about it, " \
                            " please answer the following question as if you were " \
                            "talking to a person who is blind or has low vision, answer as if you were looking at what the blind" \
                            " or low vision user wanted to know about to help them out." \


    f = open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\textFiles\\history.txt",
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

    return response



if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    # this is where we will be recording from the webcam
    # Open the webcam with explicit permission request
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows

    if not cap.isOpened():
        print("Error: Could not open webcam. Please grant camera access permission.")
        exit()

    try:

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow('Webcame Live', frame)

            key = cv2.waitKey(1)

            # 115 is the s key
            if key & 0xFF == ord('s'):

                print("System starting... ")
                # here we will be starting the real-time part of our system
                with Manager() as manager:
                    queue = manager.Queue()
                    with Pool(processes=20) as pool:

                        # current image
                        current_image = frame

                        # this is the farthest i can lower the resolution without impacting the quality of responses darastically
                        width = int(current_image.shape[1] * 0.8)
                        height = int(current_image.shape[0] * 0.8)
                        new_resolution = (width, height)

                        # Resize the image
                        img_resized = cv2.resize(current_image, new_resolution, interpolation=cv2.INTER_AREA)

                        # here we will be doing loseless compression
                        cv2.imwrite("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\current.png",
                                    img_resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

                        query = input("Question: ")

                        # here we will be calling the real-time method
                        real_time_response = real_time.real_time_system.real_time_system.real_time_test("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\current.png", pool, queue, query, model)

                        print("System: ", real_time_response)

                        # then we will be able to ask if a person wants
                        # ask further questions. If they want to then
                        # we will be calling the follow up questioning system
                        query = input("Do you have any follow up questions? (yes/no): ")

                        # here we will be starting up the follow up system
                        if query == "yes":
                            print("Loading up system... ")
                            followup_gpt4frontload("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\current.png")
                            print("System is ready!")

                            query = input("Question (if no question press n): ")

                            # here we will be starting up the follow up part of our system
                            # if there no longer are questions we will continue getting
                            # webcam footage until the user presses q again
                            while query != "n":
                                follow_up_response = get_followup_answer_gpt4frontload(query, json_reuse)

                                print("System: ", follow_up_response)

                                query = input("Question (if no question press n): ")

            if key & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Streaming has stopped")

    cap.release()

    cv2.destroyAllWindows()
    








