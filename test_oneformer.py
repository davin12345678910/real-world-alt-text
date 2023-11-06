# you need request in order to talk to any flask endpoints
# or to talk to endpoints in general
import requests


def get_oneformer():
    endpoint_url = "http://localhost:5000/get_oneformer_results"

    image_file_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\im2.png"

    image = {'image' : open(image_file_path, 'rb')}

    results = None
    # here we will make a request to the endpoint
    try:
        response = requests.post(endpoint_url, files=image)
        if response.status_code == 200:
            # print('Response from server:', response.text)
            results = response.text
        else:
            print('Error:', response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print('Error:', e)

    # now you will need to store the response back from oneformer
    return results
