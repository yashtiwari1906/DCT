import multiprocessing
import cv2
from dct import get_feature_vector_for_image 
from utils import calculate_entropy, download_cifar_dataset, save_dataframe_csv


def process_single_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    feature_vector = get_feature_vector_for_image(rgb_image)
    return feature_vector

def single_process_execution(image_paths):
    df_dict = {"image_path": [], "feature_vector": []} 

    for image_path in image_paths:
        # as of now reading image in gray scale 
        feature_vector = process_single_image(image_path)
        df_dict['image_path'].append(image_path)
        df_dict['feature_vector'].append(feature_vector)

    save_dataframe_csv(df_dict)

def multi_process_execution(image_paths):
    processes = []
    df_dict = {"image_path": [], "feature_vector": []} 
    with multiprocessing.Pool() as pool:
        results = pool.map(process_single_image, image_paths)

    for image_path, feature_vector in zip(image_paths, results):
        df_dict['image_path'].append(image_path)
        df_dict['feature_vector'].append(feature_vector)
    
    save_dataframe_csv(df_dict)


if __name__ == "__main__":
    image_paths = download_cifar_dataset(num_samples=10)
    multi_process = True  
    if multi_process:
        multi_process_execution(image_paths)
    else:
        single_process_execution(image_paths)



"""
# References:
* https://arxiv.org/html/2310.11204v2
* https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_hog.py
* https://www.researchgate.net/publication/274174155_Feature_Image_Generation_Using_Low_Mid_and_High_Frequency_Regions_for_Face_Recognition

"""