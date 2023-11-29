# Real-World-Alt-Text
## Authors: Davin Win Kyi, Jae Lee, Arvind Manivannan

## Purpose of our system:
Real-world-alt-text is a system made by our good friends at the Makeability Lab: Davin Win Kyi, Arvind Manivannan and 
JaeWook Lee. Our system helps aid blind and low vision users navigate their surrondings. There currently are solutions
for blind and low vision individuals but they lack a few attributes:
- they are hard to carry 
- they lack a human aspect to their use (conversation)
- they are not discrete
- navigating through touch isn't always the most informative
With our new system we are using a hololense and smart glasses in the future in order to have a discrete system that you
do not have to carry! We are also having a conversational approach to our device which prevents the need to navigate 
through touch on a phone screen in order to get information and the conversational approach gives a more human-like
touch to the experience. 


## Example use case of our system:
![](C:\Users\davin\PycharmProjects\real-world-alt-text_test\test-image\awesome_women_and_dogs.jpg)

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

Note: make sure you have torch and cuda toolkit installed and make sure the cuda toolkit and torch are compatible 
cuda toolkit: https://developer.nvidia.com/cuda-toolkit
torch: https://pytorch.org/get-started/locally/

### these are for the main system  
1. `git clone https://github.com/davin12345678910/real-world-alt-text.git`
2. `conda create --name real_world_alt_text-makeability python=3.11`
3. `conda activate real_world_alt_text-makeability`
4. `pip install requests`
5. `pip install openai==0.28`
6.  delete the directory that contains JsonCombiner 
7. `git clone https://github.com/davin12345678910/JsonCombiner.git`
8. `pip install shapely`
9. `pip install pillow`
10. `pip install easyocr`
11. `pip install pycocotools`
12. `pip install replicate`
13. `pip install opencv-python`
14. `pip install imantics`
15. if you get an error that says "Error: 'choices', you will need to get your own openai API KEY and set it 

### These are for the blip2 server 
1. `pip install fastapi`
2. `pip install git+https://github.com/huggingface/transformers.git`
3. `pip install uvicorn`
4. `pip install python-multipart`
5. `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`


### these are for the real-time system 
1. `conda create --name real-time python=3.11.5`
2. `conda activate real-time`
3. `pip install ultralytics`
4. `pip install openai==0.28`
5. `pip install shapely`


## How to run programs in our code 

### How to run the image summarization and follow up application 
1. `python main.py`

### blip2 server
1. `cd blip2_server`
2. `uvicorn app:app --reload`

### How to run the real time system
1. `cd real_time`
2. `cd real_time_system`
3. `python real_time_system.py`
 

## Have any Questions?
Feel free to contact me at: davin123@uw.edu and davin123@cs.washington.edu 





