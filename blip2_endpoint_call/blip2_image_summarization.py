import requests
from PIL import Image
from io import BytesIO

'''''''''
Definition: 
this code calls the blip2_endpoint_call server and return a json of the response from blip2_endpoint_call

Parameters:
Path - this is the path of the image that we want to pass into blip2_endpoint_call for captioning 

Returns:
a response from blip2 with a description of the given image that was passed in as a path 

'''
def get_blip2(path):
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

    # We will be asking what is the most you can tell me about the given image to the server
    data = {'text': "Question: What is the most you can tell me about the given image? Answer:"}

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

    # return the result from the endpoint
    return results

