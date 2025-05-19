import nibabel as nib 
import numpy as np

def load_data(path):
  #loading the data
  input_data = nib.load(path)

  #converting it into 3D numpy array (height, width, depth)
  input_data_array = input_data.get_fdata()
  return input_data, input_data_array

def find_bone_threshold_by_histogram(ct_array, bins=1000, hu_range=(-1000, 3000), gradient_window=5):
    hu_values = ct_array.flatten()
    hu_values = hu_values[np.isfinite(hu_values)]
    hu_values = np.clip(hu_values, *hu_range)

    hist, bin_edges = np.histogram(hu_values, bins=bins, range=hu_range)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Smooth histogram
    smoothed_hist = np.convolve(hist, np.ones(gradient_window)/gradient_window, mode='same')
    
    # Gradient of histogram
    grad = np.gradient(smoothed_hist)

    # Find first sharp increase above 100 HU
    for i in range(len(bin_centers)):
        if bin_centers[i] > 100 and grad[i] > 0 and smoothed_hist[i] > 100:
            return bin_centers[i]

    return 200  # Default fallback