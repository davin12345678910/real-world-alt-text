import requests
from PIL import Image
from io import BytesIO

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

    files = {'image': ('image.png', img_byte_arr, 'image/png')}

    # here we will be getting the response from the endpoint
    results = None

    # here we will make a request to the endpoint
    try:
        response = requests.post(endpoint_url, files=files)
        if response.status_code == 200:
            results = response.text
        else:
            print('Error:', response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print('Error:', e)

    return results

