import replicate
import os

'''''''''
Description: 
This method allows a user to get a description of a given image 
that is in the given image path 

Parameters:
prompt - this is the question that we will be asking to blip2_endpoint_call 

Return: 
'''
def get_blip2(prompt):

    # this is the authentication key for replicate
    os.environ["REPLICATE_API_TOKEN"] = "r8_HfRAhxwo6UJWnpdEPkLhpwC9HIRxGzn0fHb38"

    # get the output from blip2_endpoint_call
    output = replicate.run(
        "andreasjansson/blip-2:9109553e37d266369f2750e407ab95649c63eb8e13f13b1f3983ff0feb2f9ef7",
        input={"image": open("/test-image/current.png", "rb"), "question" : prompt}
    )

    # return output from blip2_endpoint_call
    return output