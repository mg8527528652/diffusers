import os 
import cv2
import numpy as np
import shutil
from tqdm import tqdm
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

if __name__ == "__main__":
    main_rename_files()
