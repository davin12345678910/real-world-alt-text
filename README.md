# Real-World-Alt-Text
## Authors: Davin Win Kyi, Jae Lee, Arvind Manivannan

## Purpose of our system:
Real-world-alt-text is a system made by our good friends at the Makeability Lab: Davin Win Kyi, Arvind Manivannan and 
Jaewook Lee. Our system helps aid blind and low vision users navigate their surrondings. There currently are solutions
for blind and low vision individuals but they lack a few attributes:
- they are hard to carry 
- they lack a human aspect to their use 
- they are not discrete
- navigating through touch isn't always the most informative
With our new system we are using a hololense and smart glasses in the future in order to have a discrete system that you
do not have to carry! We are also having a conversational approach to our device which prevents the need to navigate 
through touch on a phone screen in order to get information and the conversational approach gives a more human-like
touch to the experience. 


## Example use case of our system:
![My Image](C:\Users\davin\PycharmProjects\real-world-alt-text\test-image\current.png)
User: what is in front of me?

System: a bus and two people waiting in line

User: what is the bus number and where is the bus headed?

System: the bus number is 4604 and the bus says its headed to Bellevue

User: What are the people in line wearing?

System: The first person is wearing a black coat and a black bag and the second person is wearing
a black bag and jeans. 


## System overall architecture:

### JsonCombiner 
In this sub-directory, we have the code that allows our system to combine all of the information from 
various important cv models in order to get a json in which we can pass into gpt-4 for queries

- JsonParserTestJsons (might be removed)
  - This contains the test jsons which we will be using to test our system 
- MainJsons (might be removed)
  - this is the main json in which we will be using to give json information to our system 
- python
  - this contains the python code which we will be using the combine the information from our jsons 
- textfiles
  - history.txt: this contains previous chat information 

### miscellaneous
This folder contains files that could possibly be used for later 


### test-image:
This folder contains test images in which can be used for testing purposes 


### blip2_test.py
This file allows a user to get image summarization information from blip2


### BuildLLaVAQuery.py
This file builds a LLaVAQuery in which we will pass into LLaVA so that we can get an optimal response 
for densecaptions 

### easyocr_test.py
This file allows us to get text information for a given image along with the bounding boxes for 
for the text that is recieved 

### main.py
This file is where we will pass in all of the jsons to get a combined json and pass it into gpt-4 along with a query 
in order to get an answer from gpt-4

### oneformer.json
This is where we will be storing the output from oneformer 

### test_llava.py
This is where we will be giving a prompt and image to llava which will allows us 

### test_oneformer.py
This is where we will be running oneformer 




## Installation Steps:

(In progress... )

1. `git clone https://github.com/davin12345678910/real-world-alt-text.git`
2. `pip install requests`
3. `pip install pillow`
4. `pip install transformers`
5. `pip install torch`
6. `pip install subprocess`
7. `pip install json`
8. `pip install pycocotools.mask`
9. `pip install imantics`
10. `pip install numpy`
11. `pip install subprocess`
12. `pip install opencv-python`

## Have any Questions?
Feel free to contact me at: davin123@uw.edu and davin123@cs.washington.edu 





