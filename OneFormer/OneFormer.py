import warnings
from collections import defaultdict
from pathlib import Path
import sys

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers.models.oneformer.modeling_oneformer import OneFormerForUniversalSegmentationOutput
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from PIL import ImageOps

import openvino

import ipywidgets as widgets

sys.path.append("../utils")
# from notebook_utils import download_file

from ipywidgets import Dropdown

import matplotlib
matplotlib.use("Agg")  # disable showing figures

# you will also need to install scipy
# also use this when installing natten: https://shi-labs.com/natten/

import numpy as np

# this code is used in order to run instance segmentation
# on a given image
IR_PATH = Path("oneformer.xml")
OUTPUT_NAMES = ['class_queries_logits', 'masks_queries_logits']

processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_dinat_large")
#"oneformer_coco_swin_large"
model = OneFormerForUniversalSegmentation.from_pretrained(
    "shi-labs/oneformer_coco_dinat_large",
)
id2label = model.config.id2label

task_seq_length = processor.task_seq_length
shape = (800, 800)
dummy_input = {
    "pixel_values": torch.randn(1, 3, *shape),
    "task_inputs": torch.randn(1, task_seq_length),
    "pixel_mask": torch.randn(1, *shape),
}

model.config.torchscript = True

if not IR_PATH.exists():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = openvino.convert_model(model, example_input=dummy_input)
    openvino.save_model(model, IR_PATH, compress_to_fp16=False)

core = openvino.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)


def prepare_inputs(image: Image.Image, task: str):
    """Convert image to model input"""
    image = ImageOps.pad(image, shape)
    inputs = processor(image, [task], return_tensors="pt")
    converted = {
        'pixel_values': inputs['pixel_values'],
        'task_inputs': inputs['task_inputs']
    }
    return converted

def process_output(d):
    """Convert OpenVINO model output to HuggingFace representation for visualization"""
    hf_kwargs = {
        output_name: torch.tensor(d[output_name]) for output_name in OUTPUT_NAMES
    }

    return OneFormerForUniversalSegmentationOutput(**hf_kwargs)

# Read the model from files.
model = core.read_model(model=IR_PATH)
# Compile the model.
compiled_model = core.compile_model(model=model, device_name=device.value)

class Visualizer:
    @staticmethod
    def extract_legend(handles):
        fig = plt.figure()
        fig.legend(handles=handles, ncol=len(handles) // 20 + 1, loc='center')
        fig.tight_layout()
        return fig

    @staticmethod
    def predicted_semantic_map_to_figure(predicted_map):
        segmentation = predicted_map[0]
        # get the used color map
        viridis = plt.get_cmap('viridis', torch.max(segmentation))
        # get all the unique numbers
        labels_ids = torch.unique(segmentation).tolist()
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        ax.set_axis_off()
        handles = []
        for label_id in labels_ids:
            label = id2label[label_id]
            color = viridis(label_id)
            handles.append(mpatches.Patch(color=color, label=label))
        fig_legend = Visualizer.extract_legend(handles=handles)
        fig.tight_layout()
        return fig, fig_legend

    @staticmethod
    def predicted_instance_map_to_figure(predicted_map):
        segmentation = predicted_map[0]['segmentation']
        segments_info = predicted_map[0]['segments_info']
        # get the used color map
        viridis = plt.get_cmap('viridis', torch.max(segmentation))
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        ax.set_axis_off()
        instances_counter = defaultdict(int)
        handles = []
        # for each segment, draw its legend
        for segment in segments_info:
            segment_id = segment['id']
            segment_label_id = segment['label_id']
            segment_label = id2label[segment_label_id]
            label = f"{segment_label}-{instances_counter[segment_label_id]}"
            instances_counter[segment_label_id] += 1
            color = viridis(segment_id)
            handles.append(mpatches.Patch(color=color, label=label))

        fig_legend = Visualizer.extract_legend(handles)
        fig.tight_layout()
        return fig, fig_legend

    @staticmethod
    def predicted_panoptic_map_to_figure(predicted_map):
        segmentation = predicted_map[0]['segmentation']
        segments_info = predicted_map[0]['segments_info']
        # get the used color map
        viridis = plt.get_cmap('viridis', torch.max(segmentation))
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        ax.set_axis_off()
        instances_counter = defaultdict(int)
        handles = []
        # for each segment, draw its legend
        for segment in segments_info:
            segment_id = segment['id']
            segment_label_id = segment['label_id']
            segment_label = id2label[segment_label_id]
            label = f"{segment_label}-{instances_counter[segment_label_id]}"
            instances_counter[segment_label_id] += 1
            color = viridis(segment_id)
            handles.append(mpatches.Patch(color=color, label=label))

        fig_legend = Visualizer.extract_legend(handles)
        fig.tight_layout()
        return fig, fig_legend

def segment(img: Image.Image, task: str):
    """
    Apply segmentation on an image.

    Args:
        img: Input image. It will be resized to 800x800.
        task: String describing the segmentation task. Supported values are: "semantic", "instance" and "panoptic".
    Returns:
        Tuple[Figure, Figure]: Segmentation map and legend charts.
    """
    if img is None:
        print("Please load the image or use one from the examples list")
    inputs = prepare_inputs(img, task)
    outputs = compiled_model(inputs)
    hf_output = process_output(outputs)
    predicted_map = getattr(processor, f"post_process_{task}_segmentation")(
        hf_output, target_sizes=[img.size[::-1]]
    )
    return getattr(Visualizer, f"predicted_{task}_map_to_figure")(predicted_map)

# image = download_file("http://images.cocodataset.org/val2017/000000439180.jpg", "test_img#1.jpg")
image = Image.open("images/dog-park.jpg")

# maybe put semantic later
task = Dropdown(options=["semantic", "instance", "panoptic"], value="instance")

def stack_images_horizontally(img1: Image, img2: Image):
    res = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)), (255, 255,255))
    res.paste(img1, (0, 0))
    res.paste(img2, (img1.width, 0))
    return res

result, legend = segment(image, task.value)

result.savefig("result.jpg", bbox_inches="tight")
legend.savefig("legend.jpg", bbox_inches="tight")
result = Image.open("result.jpg")
legend = Image.open("legend.jpg")

'''''''''
res = stack_images_horizontally(result, legend)

res.show()
'''

original_image = Image.open("images/dog-park.jpg")

# Resize the original image to 800x800 pixels
original_image = original_image.resize((800, 800))

# Load the resized image
resized_image = Image.open("result.jpg")

# Resize the resized image to 800x800 pixels (if not already)
resized_image = resized_image.resize((800, 800))

# Ensure both images have the same color mode (e.g., RGBA)
original_image = original_image.convert("RGBA")
resized_image = resized_image.convert("RGBA")

# Overlay the resized image on top of the original image
# overlay = Image.alpha_composite(original_image, resized_image)

# Save or display the overlayed image
overlay = stack_images_horizontally(result, legend)
overlay.save("overlayed_image.png")



# To display the image (optional)
# overlay.show()
