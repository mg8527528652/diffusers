import os 
import cv2
import numpy as np


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
        for image in images:
            image_path = os.path.join(subfolder_path, image)
            new_image_path = os.path.join(save_folder_path, f"{subfolder}_{image}")
            os.rename(image_path, new_image_path)

def main_rename_files():
    data_folder_path = "/Users/shreyas/Desktop/data"
    save_folder_path = "/Users/shreyas/Desktop/data_renamed"
    rename_files(data_folder_path, save_folder_path)

if __name__ == "__main__":
    main_rename_files()
