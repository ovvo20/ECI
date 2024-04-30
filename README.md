# ECI

## Requirements and Installation

We recommend the requirements as follows.

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/ovvo20/ECI.git
cd ECI
```

2. Install Package
```Shell
conda create -n eci python=3.10 -y
conda activate eci
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
### Upgrade to the latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade, please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

## Model

### Pretrained Models
- [TinyLLaVA-1.5B]

### Our Models
 - [ECIMM]

## Dataset

### Relationship Data

1. To create the relationship data, we use the prompt:
```
Create a detailed list in English of all the relationships and actions involving objects depicted in the image. For each pair of objects, describe their spatial relationship using simple structures such as 'object next to object', 'object above object', 'object behind object', or 'object in front of object'. Additionally, include descriptions of actions being performed by people, animals, or changes in the environment. For instance, if the image includes a cat sitting under a tree and a bird flying toward the tree, your list should include 'cat sitting under the tree' and 'bird flying toward the tree'. Ensure that each described relationship and action between any two objects or entities is unique and clearly stated.
```

2. Data Sample:

![beach](https://github.com/ovvo20/ECI/assets/122522813/6b3a0b74-73b6-4682-98c7-eade6da3dbb8)

```
woman riding on a toy horse cart, toy horse cart on the sidewalk, bicycle parked near the beach, plastic bag on the ground near the bicycle, person walking on the beach, small buildings along the beach, flags flying above the beach, beach next to the ocean
```

### Inference Chain
1. To create this dataset, we use the nltk tool to extract the nouns and verbs from the relationship
2. Dataset format:

![talking](https://github.com/ovvo20/ECI/assets/122522813/7a955a7c-3d62-4ace-90e7-99fe4c03bff6)

```
[
    {
        "id": "2152057198",
        "image": "2152057198.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nreference: the men are in their separate homes asleep, based on the relationship between the objects in the image, the essential words and the structure of the reference, summarize a caption of the image."
            },
            {
                "from": "gpt",
                "value": "relationship: two men standing facing each other on the floor, man with red bag standing to the right of two men talking, man in dark clothing standing to the left of two men talking, group of people in the distance behind the men talking, blue tiled wall behind the group of people, pillar next to the blue tiled wall, decorative window above the blue tiled wall, ceiling above all the people in the room, scaffold on the left side of the room, light entering through the windows on the wall\nessential words: men, standing, facing, floor, man, bag, right, talking, clothing, left, group, people, distance, blue, tiled, wall, window, ceiling, room, scaffold, side, entering, windows\ncaption: the men are conversing"
            }
        ]
    },...
]
```

## Finetune

```Shell
bash finetune.sh
```

## Evaluate

```Shell
python generate.py
python evaluate.py
```

## Results

|        | Bleu_1 | Bleu_2 | Bleu_3 | Bleu_4 | METEOR | ROUGE_L | CIDEr | SPICE |
|--------|--------|--------|--------|--------|--------|---------|-------|-------|
| TIger  | 38.3   | 28.1   | 21.1   | 14.9   |        | 42.7    | 148.3 | 32.0  |
| ECIMM  | 40.3   | 30     | 22.5   | 15.8   | 19.5   | 42.8    | 152.6 | 32.7  |
