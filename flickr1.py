import json
import os
import sys
from io import StringIO
from tinyllava.model.builder import load_pretrained_model
from tinyllava.mm_utils import get_model_name_from_path
from tinyllava.eval.run_tiny_llava import eval_model

# Load JSON data
with open('infer_test.json', 'r') as file:
    data = json.load(file)

model_path = "TinyLLaVA-1.5B-finetune-i/checkpoint-450"
image_folder_path = "data/flickr30k-images/"

# Iterate over each item in the JSON data
for item in data:
    prompt = item["Ref-Cap"]
    image_id = str(item["image_id"])
    image_file = os.path.join(image_folder_path, f"{image_id}.jpg")

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": "v1",
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    # Evaluate the model and capture the output
    eval_model(args)

    # Reset stdout and extract the desired information
    sys.stdout.seek(0)
    output = sys.stdout.read().strip()
    sys.stdout = old_stdout

    # Extract the description after "\nDescription: "
    description = output.split("\ncaption: ")[-1] if "\ncaption: " in output else ""

    # Update the GT-Cap field with the extracted description
    item["GT-Cap"] = description

# Save the updated JSON data back to a new file
with open('infer_generation.json', 'w') as f:
    json.dump(data, f, indent=4)
