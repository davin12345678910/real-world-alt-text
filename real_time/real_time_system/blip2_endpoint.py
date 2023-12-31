import requests
from PIL import Image
from io import BytesIO


'''''''''
Definition:
this method allows a user to get the response from blip2 about a given image with a certain path along with information
about an object that the current image is representing 

Parameters:
path - path to the image that we will be analyzing 
object - this is the object that was detected in the image 

Returns: as much information that blip2 can find from the given image 
'''
def get_blip2(path, object):

    # this is the endpoint that we will be calling
    endpoint_url = "http://127.0.0.1:8000/blip2_predict"

    # this is the image in which we will be getting the instance segmentation for
    image_file_path = path
    image = Image.open(image_file_path)

    # Assuming `image` is your PIL Image object
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # this is the image that we will be processing
    files = {'image': ('image.png', img_byte_arr, 'image/png')}

    '''''''''
    Previous prompts:
    what are the maximum amount of details you can tell me about the given image?
    '''
    # this is the prompt that we will be sending to the server
    data = {'text': "Question: What is the most you can tell me about the " + object + " in the given image? Answer:"}

    # here we will be getting the response from the endpoint
    results = None

    # here we will make a request to the endpoint
    try:
        response = requests.post(endpoint_url, files=files, data=data)
        if response.status_code == 200:
            results = response.text
        else:
            print('Error:', response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print('Error:', e)

    return results


