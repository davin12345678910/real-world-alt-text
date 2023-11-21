import replicate
import os

'''''''''
Description: This method allows a user to get a description of a given image 
that is in the given image path 
'''
def get_blip2(prompt):

    # this is the authentication key
    os.environ["REPLICATE_API_TOKEN"] = "r8_HfRAhxwo6UJWnpdEPkLhpwC9HIRxGzn0fHb38"

    # get the output from blip2
    output = replicate.run(
        "andreasjansson/blip-2:9109553e37d266369f2750e407ab95649c63eb8e13f13b1f3983ff0feb2f9ef7",
        input={"image": open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text\\test-image\\current.png", "rb"), "question" : prompt}
    )

    return output