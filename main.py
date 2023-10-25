# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import blip2

import requests
from PIL import Image

def get_description():
    # Use a breakpoint in the code line below to debug your script.

    # this will depend on your path name
    file_path = "C:\\Users\\davin123\\PycharmProjects\\makeability_real-world-alt-text\\test-image\\king_county_buses.jpg"

    image = None

    try:
        image = Image.open(file_path).convert('RGB')
        # Now you can work with the 'image' object
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    caption = blip2.getBlip2(image)
    print(caption)

    # here we will be getting the data for RTMDet

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_description()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
