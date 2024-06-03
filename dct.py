import pandas as pd
import numpy as np 
import scipy 
from scipy.fft import dct
from utils import calculate_entropy
from zigzag_algorithm import zigzag_traversal

epsilon = 1e-10

def get_DCT_coff_matrix(nxn_image_block):

    """
    Note: I think direct grayscale image we can also use as used in the paper attached in the references

    Here we are calculating DCT coeff matrix for all channels and then taking the max coefficient as the resultant coeff for the 
    DCT matrix output of the image. inspiration took from HOG calculation reason we don't want to miss on any important frequeny information 
    present in any particular channel which is not there in any another
    """
    
    red_channel_image_block = nxn_image_block[:, :, 0]
    green_channel_image_block = nxn_image_block[:, :, 1]
    blue_channel_image_block = nxn_image_block[:, :, 2]

    image_block_dct_matrix = np.empty_like(nxn_image_block, dtype="float64")

    red_channel_dct_matrix = dct(dct(red_channel_image_block.T, norm='ortho').T, norm='ortho') + epsilon
    green_channel_dct_matrix = dct(dct(green_channel_image_block.T, norm='ortho').T, norm='ortho') + epsilon
    blue_channel_dct_matrix = dct(dct(blue_channel_image_block.T, norm='ortho').T, norm='ortho') + epsilon

    image_block_dct_matrix[:, :, 0] = red_channel_dct_matrix
    image_block_dct_matrix[:, :, 1] = green_channel_dct_matrix
    image_block_dct_matrix[:, :, 2] = blue_channel_dct_matrix

    result_dct_matrix = np.max(image_block_dct_matrix, axis = 2)

    #making DC component 0 
    result_dct_matrix[0][0] = 0 

    return result_dct_matrix
 

def get_three_diff_frequency_list(DCT_coff_matrix):
  # An adhoc way of taking out 3 refions but didn't got any reference 
  #   R1 = DCT_coff_matrix[:4, :4]  # LF region
  #   R2 = DCT_coff_matrix[4:6, 4:6]  # MF region
  #   R3 = DCT_coff_matrix[6:, 6:]  # HF region

  traversed_output = zigzag_traversal(DCT_coff_matrix)
  length = len(traversed_output)
  # divide it in 3 equal parts ref - [3]
  R1 = traversed_output[:length // 3]
  R2 = traversed_output[length // 3 : (2*length)//3]
  R3 = traversed_output[(2*length)//3:]

  return R1, R2, R3

def get_statiscal_features_from_frequency_list(frequency_list):
  mean = np.mean(frequency_list)
  variance = np.var(frequency_list)
  kurtosis = scipy.stats.kurtosis(frequency_list)

  if np.isnan(kurtosis) or np.isinf(kurtosis):
    kurtosis = 0                                            ############################################
  skewness = scipy.stats.skew(frequency_list)               # what to impute in skewness & kurtosis    #
  if np.isnan(skewness) or np.isinf(skewness):              # when freq_list is 0 becuase of DCT coeff #
    skewness = 0                                            # is 0 (image must be smooth there)        #
  entropy = calculate_entropy(frequency_list)               ############################################
  energy = np.sum(np.square(frequency_list))

  return mean, variance, kurtosis, skewness, entropy, energy 


def get_features_for_nxn_block(nxn_image_block):
  DCT_coff_matrix = get_DCT_coff_matrix(nxn_image_block)
  # DCT_coff_matrix += epsilon
  low_frequency_list, middle_frequency_list, high_frequency_list = get_three_diff_frequency_list(DCT_coff_matrix)

  region_one_stats_vector = get_statiscal_features_from_frequency_list(low_frequency_list)
  region_two_stats_vector = get_statiscal_features_from_frequency_list(middle_frequency_list)
  region_three_stats_vector = get_statiscal_features_from_frequency_list(high_frequency_list)

  final_vector = region_one_stats_vector + region_two_stats_vector + region_three_stats_vector

  return final_vector 

def get_nxn_blocks(image):
  height, width, channels = image.shape
  assert width % 8 == 0 and height % 8 == 0, "resize the image width and height into multiple of 8"
  assert channels == 3, "image is not in RGB or BGR format check it once again"

  blocks = [] 

  for h in range(0, height, 8):
    for w in range(0, width, 8):
      image_8x8 = image[h:h+8, w:w+8, :]
      blocks.append(image_8x8)

  return blocks  
  
def get_feature_vector_for_image(image):
  image_blocks = get_nxn_blocks(image)

  feature_vectors_list = [] 

  for image_block in image_blocks:
    feature_vector = get_features_for_nxn_block(image_block)
    feature_vectors_list.append(feature_vector)

  image_feature_vector = np.mean(feature_vectors_list, axis = 0)

  return image_feature_vector


