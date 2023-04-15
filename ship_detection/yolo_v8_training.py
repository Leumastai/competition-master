

import os
import glob
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

###
# login hugging face cli
# login wandb
###


### Download datasets
#!git lfs install
#!git clone https://huggingface.co/datasets/datadrivenscience/ship-detection


# Define necessary directories
ROOT_DIR = os.path.join(os.getcwd(), 'ship-detection')

train_dir = os.path.join(ROOT_DIR, "train")
test_dir = os.path.join(ROOT_DIR, "test")

## get all images from each dirs
test_imgs = glob.glob(train_dir + "/*.png")
test_imgs = glob.glob(test_dir + "/*.png")

train_metadata = os.path.join(train_dir, 'metadata.jsonl') 
test_metadata = os.path.join(test_dir, 'metadata.jsonl')



## convert image metadata to json files

# Train data
with open(train_metadata, 'r+') as j_files:
  j_list = list(j_files)
  
bboxes = []
filename = []
categories = []

for j_fle in j_list:
  rslt = json.loads(j_fle)

  bboxes.append(rslt['objects']['bbox'])
  filename.append(rslt['file_name'])
  categories.append(rslt['objects']['categories'])


train_df = pd.DataFrame(
    {
        "filenames": filename,
        "bboxes": bboxes,
        "categories": categories,
    })

train_df = train_df.explode(['bboxes', 'categories']).reset_index(drop=True)
train_df.head()