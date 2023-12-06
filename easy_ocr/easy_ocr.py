import easyocr
import time

'''''''''''
Descriptions: 
this method allows a user to get all text information from easy_ocr 

Returns:
gives all of the text information in a given image 
'''
def get_easryocr():

    # we will have english as our default language
    reader = easyocr.Reader(['en'])

    # gives the text output of the current image
    result = reader.readtext('C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\current.png')
    return result