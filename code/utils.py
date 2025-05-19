import nibabel as nib 
import numpy as np
import matplotlib.pyplot as plt

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

def visulaize_data(data, ax, title, cmap='gray', axis=1):
  # compute middle index
  mid = data.shape[axis] // 2

  # extract the 2D slice
  if axis == 0:
      slice_2d = data[mid, :, :]
  elif axis == 1:
      slice_2d = data[:, mid, :]
  elif axis == 2:
      slice_2d = data[:, :, mid]
  else:
      raise ValueError("Axis must be 0, 1, or 2.")
  
  ax.imshow(slice_2d.T,
            origin='lower',
            cmap=cmap,
            aspect='equal')
  ax.set_title(f'Mid‐slice (axis {axis}) of {title}')
  ax.axis('off')
