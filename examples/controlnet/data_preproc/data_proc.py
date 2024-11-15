import os 
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value
import json
def rename_files(data_folder_path, save_folder_path):
    '''
    inputs:
        data_folder_path: folder containing various subfolders, in which images are stored
    Returns:
        None
    This function saves the images in the data folder to the save folder. renaming is done as: PARENT_FOLDER_NAME_IMAGE_NAME.jpg
    '''
    subfolders = os.listdir(data_folder_path)
    for subfolder in subfolders:
        subfolder_path = os.path.join(data_folder_path, subfolder)
        images = os.listdir(subfolder_path)
        for image in tqdm(images):
            try:
                # if image exists, continue 
                if not os.path.exists(os.path.join(save_folder_path, f"{subfolder}_{image}")):
                    image_path = os.path.join(subfolder_path, image)
                    new_image_path = os.path.join(save_folder_path, f"{subfolder}_{image}")
                    shutil.copy(image_path, new_image_path)
            except Exception as e:
                print(f"Error processing image {image}: {e}")
                continue

def main_rename_files():
    images_folder_path = "/home/ubuntu/Desktop/mayank_gaur/freepik/freepik_masks"
    alpha_folder_path = "/home/ubuntu/Desktop/mayank_gaur/freepik/freepik_masks"

    save_folder_path = "/home/ubuntu/Desktop/mayank_gaur/freepik/freepik_masks_renamed"
    save_alpha_folder_path = "/home/ubuntu/Desktop/mayank_gaur/freepik/freepik_alphas_renamed"
    save_images_folder_path = "/home/ubuntu/Desktop/mayank_gaur/freepik/freepik_images_renamed"

    os.makedirs(save_folder_path, exist_ok=True)
    os.makedirs(save_alpha_folder_path, exist_ok=True)
    os.makedirs(save_images_folder_path, exist_ok=True)

    rename_files(alpha_folder_path, save_alpha_folder_path)
    rename_files(images_folder_path, save_images_folder_path)



def create_dataset(image_path, condition_image_path, prompt_json_path, save_path):
    image_names = os.listdir(condition_image_path)
    print(len(image_names))

    def read_json(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)["bg_label"]

    def gen_examples():
        for i in range(len(image_names)):
            yield {
                "image": {"path": image_path + '/' + image_names[i].split(".")[0] + ".jpg"},
                "condtioning_image": {"path": condition_image_path + '/' + image_names[i]},
                "text": read_json(prompt_json_path + '/' + image_names[i].split(".")[0] + ".json")
            }

    train_dataset = Dataset.from_generator(
        gen_examples,
        features=Features(
            image=ImageFeature(),
            condtioning_image=ImageFeature(),
            text=Value("string"),
        ),
        num_proc=1,
    )
    train_dataset.info.features
    train_dataset.save_to_disk(save_path)

def main_create_dataset():
    image_path = "/home/ubuntu/projects/python/deepak/controlnet/data/data_after_preproc/input_jpg"
    condition_image_path = "/home/ubuntu/projects/python/deepak/controlnet/data/data_after_preproc/input_mm_skl_rgbgar"
    prompt_json_path = "/home/ubuntu/projects/python/deepak/controlnet/data/data_after_preproc/prompt_json"
    save_path = "/home/ubuntu/projects/python/deepak/controlnet/data/data_after_preproc/train_set_xl_mm_skl_rgbgar"
    create_dataset(image_path, condition_image_path, prompt_json_path, save_path)


def folder_diff(folder1, folder2):
    # if folder1 is a path to txt, get file names from it, else get file names from folder1
    if folder1.endswith(".txt"):
        with open(folder1, "r") as f:
            folder1_files = [line.strip() for line in f.readlines()]
    else:
        folder1_files = os.listdir(folder1)
    if folder2.endswith(".txt"):
        with open(folder2, "r") as f:
            folder2_files = [line.strip() for line in f.readlines()]
    else:
        folder2_files = os.listdir(folder2)
    folder1_files = [i.split(".")[0] for i in folder1_files]
    folder2_files = [i.split(".")[0] for i in folder2_files]
    return set(folder1_files) & set(folder2_files)

def remove_wrong_mask_files(txts_path, dataset_path, save_path):
    folder_ids = [i.split(".")[0] for i in os.listdir(txts_path)]
    for folder_id in folder_ids:
        txt_path = os.path.join(txts_path, folder_id + ".txt")
        folder_path = os.path.join(dataset_path, folder_id)
        common_files = folder_diff(folder_path, txt_path)
        for file in common_files:
            if save_path.endswith('images'):
                shutil.copy(os.path.join(folder_path, file + ".jpg"), os.path.join(save_path, file + ".jpg"))
            else:
                shutil.copy(os.path.join(folder_path, file + ".png"), os.path.join(save_path, file + ".png"))
            print(file)


if __name__ == "__main__":
    # main_rename_files()
    txts_path = r'/home/ubuntu/Desktop/mayank_gaur/freepik/non_yellow_files'
    dataset_images_path = r'/home/ubuntu/Desktop/mayank_gaur/freepik/freepik_imgs'
    dataset_alpha_path = r'/home/ubuntu/Desktop/mayank_gaur/freepik/freepik_masks'
    save_path = r'/home/ubuntu/Desktop/mayank_gaur/freepik/freepik_wrong_files'
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'alphas'), exist_ok=True)
    remove_wrong_mask_files(txts_path, dataset_images_path, os.path.join(save_path, 'images'))
    remove_wrong_mask_files(txts_path, dataset_alpha_path, os.path.join(save_path, 'alphas'))
    # main_create_dataset() 
