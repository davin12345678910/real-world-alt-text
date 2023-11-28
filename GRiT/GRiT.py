# you need request in order to talk to any flask endpoints
# or to talk to endpoints in general
import requests

'''''''''''''''
Description: 
this method will allow a user to call an endpoint for oneformer
where a user can pass in an image and get the instance segmentation of the image

Returns:
gives a text json of all of the objects in the current image 
'''
def get_grit():

    # this is the endpoint that we will be calling
    endpoint_url = "http://localhost:5001/get_grit_results"

    # this is the image in which we will be getting the instance segmentation for
    image_file_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\current.png"

    # this is the image that we will be passing in
    image = {'image' : open(image_file_path, 'rb')}

    # where we will store the result from the endpoint
    results = None

    # here we will make a request to the endpoint
    try:
        response = requests.post(endpoint_url, files=image)
        if response.status_code == 200:
            results = response.text
        else:
            print('Error:', response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print('Error:', e)

    # return the response back
    return results

