import replicate
import os

'''''''''''
Definition: allows a user to pass in a prompt for the given image

Parameters:
- prompt: this is the prompt that we will be giing to LLaVA
'''
def get_llava(prompt):

    # we need this so that we can use the replicate api
    os.environ["REPLICATE_API_TOKEN"] = "r8_HfRAhxwo6UJWnpdEPkLhpwC9HIRxGzn0fHb38"

    # this is the image that we will be analyzing
    file_path = "/test-image(2)/current.png"
    image = open(file_path, 'rb')

    # this is where we wll run the api to get the output
    output = replicate.run(
        "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
        input={"image": image, "prompt" : prompt, "temperature" : 0.01, "top_p" : 0, "max_tokens" : 5000}
    )

    # here we will parse the response
    result = ""
    for item in output:
        result = result + " " + str(item)

    if "{" not in result:
        return result
    else:
        index = result.index('{')
        final_result = result[index:]
        # print("LLaVA: ", final_result)
        return final_result
