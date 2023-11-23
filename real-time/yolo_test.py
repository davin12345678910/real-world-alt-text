from ultralytics import YOLO

import requests
# from transformers import AutoProcessor, Blip2Processor, Blip2ForConditionalGeneration
import torch
# import requests
from PIL import Image
import os
import multiprocessing
# import blip2_test
import blip2_endpoint
# import openai
import time
# import replicate
from multiprocessing import Pool
import GPT4

# import webbrowser

# from pipeline.cloud.pipelines import run_pipeline

''''''''''
def process_images_in_parallel(pool, image_tasks):
    # Use the provided pool to process tasks
    results = pool.map(get_blip2_response, image_tasks)
    return results
'''

# here we will be making a method that will call gpt4
def get_gpt4_response(path, prompt, key, queue):
    response = GPT4.get_gpt4(path, prompt, key)
    queue.put(response)

def get_blip2_response(path, query, queue):
    # replicate api
    '''''''''
    path, name = args
    # Set the API token environment variable (move this to a more global scope if needed)
    os.environ["REPLICATE_API_TOKEN"] = "r8_HfRAhxwo6UJWnpdEPkLhpwC9HIRxGzn0fHb38"

    # Run the BLIP-2 model
    output = replicate.run(
        "andreasjansson/blip-2:9109553e37d266369f2750e407ab95649c63eb8e13f13b1f3983ff0feb2f9ef7",
        input={"image": open(path, "rb"), "question": "what is in the given image"}
    )

    return {"object_name": name, "text": output}
    '''

    response = blip2_endpoint.get_blip2(path, query)

    queue.put(response)


def real_time_test(path):

    start_time = time.time()
    model = YOLO('yolov8n.pt')

    results = model(
        [path])

    boxes = None
    classes = None

    for key in results:
        boxes = key.boxes.xyxy.tolist()
        classes = key.boxes.cls.tolist()

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

    yolo_results = {}
    index = 0
    for box in boxes:
        yolo_results[index] = {'bbox': [round(float(box[0]), 2), round(float(box[1]), 2), round(float(box[2]), 2),
                                        round(float(box[3]), 2)]}

        image = Image.open(path)

        crop_area = (int(round(float(box[0]), 2)), int(round(float(box[1]), 2)), int(round(float(box[2]), 2)),
                                        int(round(float(box[3]), 2)))

        cropped_image = image.crop(crop_area)

        cropped_image.save("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\cropped_imgs\\" + "cropped_im" + str(index) + ".png")

        index = index + 1

    index = 0
    count_names = {}
    for c in classes:
        if idToObject[c] in count_names:
            yolo_results[index]["name"] = idToObject[c] + str(count_names[idToObject[c]] + 1)
            count_names[idToObject[c]] = count_names[idToObject[c]] + 1
        else:
            yolo_results[index]["name"] = idToObject[c] + str(1)
            count_names[idToObject[c]] = 1
        index = index + 1

    # print(yolo_results)

    yolo_list = []

    for object in yolo_results:
        yolo_list.append(yolo_results[object])


    # ask blip2 for a response that includes all of the objects in an image
    prompt = "Question: What can you tell me overall about each of these objects in the given image: "

    index = 0

    objects = ""

    for name in count_names:
        if index == 0:
            if count_names[name] == 1:
                objects += name + " "
            else:
                objects += str(count_names[name]) + " " + name + "s "
        else:
            if count_names[name] == 1:
                objects += " and " + name
            else:
                objects += " and " + str(count_names[name]) + " " + name + "s"

        index = index + 1

    prompt += objects + "? Also, can you tell me about the colors that are present on each of the following objects in the image: " + objects + "? Also, what are each of the following objects in the image doing: " + objects + "? Answer:"

    # print(prompt)

    # now we will pass the following into blip2
    # response = get_blip2_response(path, prompt)

    # calling gpt4:
    api_keys = ["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",
                "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",
                "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",
                "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",
                "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",
                "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",
                "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",
                "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",
                "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",
                "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"
                ]

    processes = []

    path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\cropped_imgs\\"
    queue = multiprocessing.Queue()
    for i in range(len(yolo_list)):
        current = yolo_results[i]
        name = current["name"]
        process = multiprocessing.Process(target=get_blip2_response,
                                          args=(path + "cropped_im" + str(i) + ".png", "What is in the image?", queue))
        processes.append(process)
        print("Started Process!")
        process.start()

    for process in processes:
        process.join()
        print("Joined process!")

    print("All processes complete!")

    end_time_first = time.time()

    elapsed_time_first = end_time_first - start_time

    print("Elapsed_time_first: ", elapsed_time_first)

    # here we will gather all of the output
    # data_list = []
    #while not queue.empty():
    #    data_list.append(queue.get())

    # from here work on the combination of all the information and then
    # pass the info into gpt-4 turbo for a response given a query
    '''''''''
    prompt = "Given this information about the given image: " + str(blip2_results) + ", answer the following question: " + query

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
    '''

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Elapsed_time: ", elapsed_time)

    # print(response)


# here we will be going through each of the images in the test_images
if __name__=='__main__':
    # here we will be getting multiple images, and then we will be testing them out
    directory_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\"

    file_list = os.listdir(directory_path)

    # with Pool(processes=10) as pool:
    for file in file_list:

        real_time_test(directory_path + file)






































'''''''''
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

print(results)

path = model.export(format="onnx")  # export the model to ONNX format
'''

'''''''''
import torch

model = torch.hub.load('ultralytrics/yolov8', 'yolov8s', pretrained=True)

imgs = ['https://ultralytics.com/images/bus.jpg']

results = model(imgs)

results.print()
results.show()
'''

# here we need to crop the image, and save it into an image
# then we can later pass each of these images into blip2

# here we are going to get cropped parts of the image and
# we are going to be saving them into a cropped folder
'''''''''
image_tasks = [("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\cropped_imgs\\" + "cropped_im" + str(i) + ".png", yolo_results[i]["name"]) for i in range(len(yolo_list))]

blip2_results = process_images_in_parallel(pool, image_tasks)
'''

'''''''''''
# now we will try to run a bunch of processes in parallel for blip2
processes = []

path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\real-time\\cropped_imgs\\"
queue = multiprocessing.Queue()
for i in range(len(yolo_list)):
    current = yolo_results[i]
    name = current["name"]
    process = multiprocessing.Process(target=get_blip2_response,
                                      args=(path + "cropped_im" + str(i) + ".png", queue, name))
    processes.append(process)
    print("Started Process!")
    process.start()

for process in processes:
    process.join()
    print("Joined process!")

print("All processes complete!")

end_time_first = time.time()

elapsed_time_first = end_time_first - start_time

print("Elapsed_time_first: ", elapsed_time_first)
'''




