import easyocr
import time

'''''''''''
Descriptions: this method allows a user to get all ocr information that is needed
'''
def get_easryocr():
    # start_time = time.time()
    reader = easyocr.Reader(['en'])
    result = reader.readtext('C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\current.png')
    # end_time = time.time()

    # print("OCR: ", result)

    # elapsed_time = end_time - start_time

    # print("Elapsed_time: ", elapsed_time)
    return result