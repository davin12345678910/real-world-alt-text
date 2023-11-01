import blip2
import requests
from PIL import Image
import llava.llava.serve.cli
import multiprocessing as mp


# here we will be storing the results of each of the methods
# in which we will be working with
results_oneformer = None
results_rtmdet = None
results_GRiT = None
results_ocr = None
results_LLaVA = None
results_blip2 = None


'''
Description: this method allows a user to recieve information from RTMDet
'''
def run_oneformer():
    global results_oneformer
    results_oneformer = "one former ran"

def run_rtmdet(img):
    global results_rtmdet
    results_rtmdet = "rtm det ran"

# LLaVA works now!
def run_LLaVA(img, prompt):
    # here we will be calling cli's get_llava method
    global results_LLaVA
    results_LLaVA = "LLaVA ran"
    # llava.llava.serve.cli.get_LLaVA(prompt, img)

def run_GRiT(img):
    global results_GRiT
    results_GRiT = "GRiT ran"

def run_blip2(img):
    global results_blip2
    results_blip2 = "blip2 ran"
    #caption = blip2.getBlip2(img)

def run_ocr(img):
    global results_ocr
    results_ocr = "ocr ran"



def get_followup(img):

    # here we will be doing everything in parallel
    # when you want to include parameters, have args=()
    process1 = mp.Process(target=run_oneformer)
    process2 = mp.Process(target=run_rtmdet)
    process3 = mp.Process(target=run_LLaVA)
    process4 = mp.Process(target=run_GRiT)
    process5 = mp.Process(target=run_blip2)
    process6 = mp.Process(target=run_ocr)

    process1.run()
    process2.run()
    process3.run()
    process4.run()
    process5.run()
    process6.run()

    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
    process6.join()


    # here we need to call the combine method



    return ""



''''''''''
This is where we will get 
'''
def get_summarization(prompt, img):
    print("Hello")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

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

