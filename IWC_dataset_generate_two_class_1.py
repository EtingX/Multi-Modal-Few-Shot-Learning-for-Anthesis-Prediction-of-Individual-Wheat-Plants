import os
import random
import numpy as np
import pandas as pd
import cv2
import albumentations as A

'''
    This script is designed to create datasets by randomly selecting pairs of images, along with their corresponding 
    weather data, based on annotations in the image filenames. Each sample in the dataset consists of two images and 
    their respective weather data, which include actual weather conditions for 90 days and 7-day weather forecast 
    information, focusing on minimum temperature, maximum temperature, solar exposure, rainfall, photo degree, and 
    cumulative photo degree. 
'''

def get_random_augmentations(height, width):
    '''
        Generate a random set of image augmentations. If a random chance is met, also add a
        RandomResizedCrop augmentation.

    :param height: image height
    :param width: image width
    :return: augmentation image

    '''

    augmentations = [
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
        A.RGBShift(),
        A.HueSaturationValue(),
        A.Blur(blur_limit=7),
        A.MedianBlur(blur_limit=7),
        A.GaussNoise()
    ]

    num_augmentations = random.randint(0, len(augmentations))
    chosen_augmentations = random.sample(augmentations, num_augmentations)


    # Randomly decide whether to add RandomResizedCrop
    rand_num = random.random()
    if rand_num < 0.1:
        scale_min = 0.95
        scale_max = 1.0
        chosen_augmentations.append(A.RandomResizedCrop(height, width, scale=(scale_min, scale_max)))

    return A.Compose(chosen_augmentations)


def apply_augmentations(image):
    augmentations = get_random_augmentations(image.shape[0], image.shape[1])
    augmented_image = augmentations(image=image)['image']
    return augmented_image


def load_weather_data(weather_folder, date):
    '''
    Load weather data for a specific date from an Excel file.
    '''
    weather_file = os.path.join(weather_folder, date + '.xlsx')
    if os.path.exists(weather_file):
        weather_data = pd.read_excel(weather_file)

        # Remove the first column
        weather_data = weather_data.iloc[:, 1:]

        return weather_data
    else:
        return None


def read_image_as_array(image_folder, image_file):
    # Load an image file as a numpy array.
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    return image


def random_sample_and_label(image_files, image_folder, weather_folder, save_dir, apply_aug=True):
    '''
    Create a dataset sample by randomly selecting two images and their weather data.
    The label is determined by comparing their flowering dates.
    '''
    sample1_file = random.choice(image_files)
    sample1_info = sample1_file.split('_')
    sample1_blooming_date = int(sample1_info[0])

    # Read the first sample's image and weather data
    image_data1 = read_image_as_array(image_folder, sample1_file)
    weather_data1 = load_weather_data(weather_folder, sample1_info[1])

    # 生成随机数
    rand1, rand2 = random.random(), random.random()

    # sample 2
    if rand1 < 0.20:
        if rand2 > 0.05:
            #  same flowering time but image different ID
            condition = lambda f: f.startswith(sample1_info[0]) and f.split('_')[2] != sample1_info[2]
        else:
            #  same flowering time and same image ID
            condition = lambda f: f.startswith(sample1_info[0]) and f.split('_')[2] == sample1_info[2]
    else:
        if rand2 > 0.05:
            # different flowering time and image different ID
            condition = lambda f: not f.startswith(sample1_info[0]) and f.split('_')[2] != sample1_info[2]
        else:
            # different flowering time and image same ID
            condition = lambda f: not f.startswith(sample1_info[0]) and f.split('_')[2] == sample1_info[2]

    # Filter the files based on the condition
    sample2_files = [f for f in image_files if condition(f)]
    if not sample2_files:
        return None, None, None, None

    sample2_file = random.choice(sample2_files)
    sample2_info = sample2_file.split('_')
    sample2_blooming_date = int(sample2_info[0])

    # Read the second sample's image and weather data
    image_data2 = read_image_as_array(image_folder, sample2_file)
    weather_data2 = load_weather_data(weather_folder, sample2_info[1])

    # Apply image augmentations if enabled
    if apply_aug:
        print('Image augmentation processing')
        image_data1 = apply_augmentations(image_data1)
        image_data2 = apply_augmentations(image_data2)

    # Create the dataset
    dataset = [[image_data1, weather_data1], [image_data2, weather_data2]]

    # Determine the label based on the blooming dates
    label = '1_' if sample1_blooming_date < sample2_blooming_date else '0_'
    check_label = 1 if sample1_blooming_date < sample2_blooming_date else 0
    # Prepare the dataset and label
    if label == '1_':
        save_dir = os.path.join(save_dir, '1')
    else:
        save_dir = os.path.join(save_dir, '0')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the dataset
    sample1_filename = os.path.splitext(sample1_file)[0]
    sample2_filename = os.path.splitext(sample2_file)[0]
    save_filename = f"{label}{sample1_filename}_{sample2_filename}.npy"
    save_filename = os.path.join(save_dir, save_filename)
    np.save(save_filename, np.array(dataset, dtype=object))

    return save_filename, sample1_file, sample2_file, check_label


def generate_datasets(image_folder, weather_folder, save_dir, num_datasets=120):
    # Generate a specified number of dataset samples.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_files = os.listdir(image_folder)
    total_images = len(image_files) - 1
    generated_datasets = 0
    used_images = set()

    num_label_0 = 0
    num_label_1 = 0
    max_num_label_0 = int(num_datasets/2)+5

    max_num_label_1 = int(num_datasets/2)+5

    while generated_datasets < int(num_datasets):
        # Ensuring each image is used at least once
        if len(used_images) < total_images:
            sample_file = [f for f in image_files if f not in used_images]
            label, sample_1, sample_2, check_label = random_sample_and_label(sample_file, image_folder, weather_folder, save_dir)
            if label != None:
                used_images.add(sample_1)
                used_images.add(sample_2)

                if check_label == 1:
                    if num_label_1 < max_num_label_1:
                        generated_datasets += 1
                        print(f"Different Dataset {generated_datasets + 1} generated: {os.path.basename(label)}")
                        num_label_1 = num_label_1 + 1
                    else:
                        os.remove(label)
                        print('Max 1')
                elif check_label == 0:
                    if num_label_0 < max_num_label_0:
                        generated_datasets += 1
                        print(f"Different Dataset {generated_datasets + 1} generated: {os.path.basename(label)}")
                        num_label_0 = num_label_0 + 1
                    else:
                        os.remove(label)
                        print('Max 0')

        else:
            sample_file = image_files
            label, sample_1, sample_2, check_label = random_sample_and_label(sample_file, image_folder, weather_folder, save_dir)

            if check_label == 1:
                num_label_1 = num_label_1 + 1
                if num_label_1 <= max_num_label_1:
                    generated_datasets += 1
                    print(f"Dataset {generated_datasets + 1} generated: {os.path.basename(label)}")
                else:
                    os.remove(label)
                    print('Max 1')
            elif check_label == 0:
                num_label_0 = num_label_0 + 1
                if num_label_0 <= max_num_label_0:
                    generated_datasets += 1
                    print(f"Dataset {generated_datasets + 1} generated: {os.path.basename(label)}")
                else:
                    os.remove(label)
                    print('Max 0')

    print(f"All {num_datasets} datasets generated.")


# example
home_dir = 'I:/wheat project/few shot flowering project/few shot data set'
image_folder_path = os.path.join(home_dir,'select data/TPA select 2 single label 2023\selected_TPA_ID_photos_20')
weather_folder_path = 'I:/wheat project/few shot flowering project/few shot data set/weather'
save_dir = 'I:/wheat project/few shot flowering project\paper/TPA 2 less dataset'

generate_datasets(image_folder_path, weather_folder_path, save_dir)
