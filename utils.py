import torch
import torchvision
from torchvision import transforms
import numpy as np 
import pandas as pd 


def calculate_entropy(data):
    hist, _ = np.histogram(data, bins=256, density=True)
    
    hist_norm = hist / np.sum(hist)
    
    # Compute entropy
    epsilon = 1e-10  # Small value to avoid log(0)
    entropy = -np.sum(hist_norm * np.log2(hist_norm + epsilon))
    
    return entropy

def save_dataframe_csv(df_dict):
    features = [
        "low_frequency_mean", "low_frequency_variance", "low_frequency_kurtosis", "low_frequency_skewness", "low_frequency_entropy", "low_frequency_energy",
        "middle_frequency_mean", "middle_frequency_variance", "middle_frequency_kurtosis", "middle_frequency_skewness", "middle_frequency_entropy", "middle_frequency_energy",
        "high_frequency_mean", "high_frequency_variance", "high_frequency_kurtosis", "high_frequency_skewness", "high_frequency_entropy", "high_frequency_energy",
     ] 

    result_df_dict = {"image_paths": []}
    
    for image_path, feature_vector in zip(df_dict['image_path'], df_dict['feature_vector']):
        result_df_dict['image_paths'].append(image_path)
        for feature, feature_value in zip(features, feature_vector):
            result_df_dict[feature] = [feature_value] + result_df_dict.get(feature, []) 
    
    result_df = pd.DataFrame(result_df_dict)
    result_df.to_csv("resultant_dataframe.csv", index = False)
    print(f"DataFrame saved successfully!!!")

def download_cifar_dataset(num_samples = 10):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
    ])

    # # Download CIFAR-10 dataset
    cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    image_paths = [] 
    # Extract and save 10 images
    for i in range(num_samples):
        image, label = cifar10_train[i]
        path = f'cifar10_image_{i}.png'
        torchvision.utils.save_image(image, path)
        image_paths.append(path)

    print("Images downloaded successfully.")
    return image_paths 