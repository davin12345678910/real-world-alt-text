import io
import json
from mrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes, VisualFeatureTypes
from PIL import Image, ImageDraw, ImageFont

API_KEY = "46fc4a686ee24a468cf27e135ba73336"
ENDPOINT = "https://davin123.cognitiveservices.azure.com/"

cv_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

