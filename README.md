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

User: How many dogs are in front of me?
System: There are four dogs in front of me

User: What colors are the dogs?
System: The colors of the dogs are black and white, white, brown and white, and white with brown 

User: Are there any people in the image ?
System: Yes, there are people in the image; they are sitting on a bench 



## System overall architecture:

### blip2_endpoint_call
This is the code that allows an individual to call the blip2 endpoint that we are running 

### blip2_server
This code is for the blip2 server which we will be running that can process calls concurrently

### easy_ocr
This code call easy_ocr which is a model that will help us get text information for a given image 

### GPT4
This code is used for our system that directly calls GPT4 vision by passing in an image and prompt directly 

### GPT4Frontload 
This code is used for our system that gets a json from gpt4 vision and uses that json for later queries 

### GPT4Main
This is the code we use to get gpt4 information for our main system with 4 models 

### GRiT
This is the code used to get dense captioning information for our main system with 4 models 

### JsonCombiner
This is the code that we use in order to combine the data from 4 models in our main system with 4 models 

### miscellaneous
This is the code that we might possibly use in the future 

### oneformer
This is the code used for the instance segmentation for our main system with 4 models 

### real-time
This is the code used for our real-time system 

### test-image
These are some test images that we use for testing 

### test-image(2)
This is a second batch of images that we might use for later 

### testFiles
This contains the text files specifically the history file that contains all of the questions that a person has asked 

### current.png
This is the current png that we will be analyzing 

### main.py
This is where we will be runnning all of the code in our system 






## Installation Steps:

Note: make sure you have torch and cuda toolkit installed and make sure the cuda toolkit and torch are compatible 
cuda toolkit: https://developer.nvidia.com/cuda-toolkit
torch: https://pytorch.org/get-started/locally/

### these are for the main system (image summarization + ) 
1. `git clone https://github.com/davin12345678910/real-world-alt-text.git`
2. `conda create --name real_world_alt_text-makeability python=3.11`
3. `conda activate real_world_alt_text-makeability`
4. `pip install -r requirements.txt`
6.  delete the directory that contains JsonCombiner 
7. `git clone https://github.com/davin12345678910/JsonCombiner.git`
8. `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
9. if you get an error that says "Error: 'choices', you will need to get your own openai API KEY and set it 

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


### How to use mainsys.py
1. do the same steps for blip2 server
2. in a seperate terminal do `conda activate real-time`
3. Then you can start mainsys.py
 

## Have any Questions?
Feel free to contact me at: davin123@uw.edu and davin123@cs.washington.edu 





