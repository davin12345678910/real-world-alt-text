import easyocr
import time

'''''''''''
Descriptions: this method allows a user to get all ocr information that is needed
'''
def get_easryocr():
    reader = easyocr.Reader(['en'])
    result = reader.readtext('C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\current.png')
    return result