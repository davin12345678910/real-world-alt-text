import easyocr
import time

'''''''''''
Descriptions: 
this method allows a user to get all easy_ocr information that is needed

Returns:
gives a result json with a list of the strings that were detected in a given image 
'''
def get_easryocr():

    # give the languages you want the model to detect
    reader = easyocr.Reader(['en'])

    # gives the text output of the current image
    result = reader.readtext('C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\current.png')
    return result