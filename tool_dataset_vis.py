import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def visualize_dataset(npy_path):
    dataset = np.load(npy_path, allow_pickle=True)

    sample1_image_data = dataset[0][0]
    sample1_weather_data = dataset[0][1]

    sample1_image_data_rgb = cv2.cvtColor(sample1_image_data, cv2.COLOR_BGR2RGB)

    plt.imshow(sample1_image_data_rgb)
    plt.title("Sample 1 image")
    plt.show()

    print("Sample 1 weather dataframe")
    print(pd.DataFrame(sample1_weather_data))

    if len(dataset) > 1 and dataset[1][0] is not None:
        sample2_image_data = dataset[1][0]
        sample2_weather_data = dataset[1][1]

        sample2_image_data_rgb = cv2.cvtColor(sample2_image_data, cv2.COLOR_BGR2RGB)

        plt.imshow(sample2_image_data_rgb)
        plt.title("Sample 2 image")
        plt.show()

        print("Sample 2 weather dataframe")
        print(pd.DataFrame(sample2_weather_data))


visualize_dataset("0_13_2023-08-30_BC-ID76_IMG_5759_30_11_2023-09-01_BC-ID76_IMG_8484_25.npy")