import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage
from utils import load_data,find_bone_threshold_by_histogram,visulaize_data

def threshold_and_label(ct_data, hu_threshold_value=150):
  # Create a binary mask where bone voxels are 1, others 0
  bone_mask = (ct_data >= hu_threshold_value)
  labels, num_labels = ndimage.label(bone_mask)
  return bone_mask,labels, num_labels

def split_femur_tibia_by_slice(bone_mask, axis=2, cut_frac=0.5):
  #determine cut index
  size = bone_mask.shape[axis]
  cut = int(size * cut_frac)

  #initializing empty mask
  femur_mask = np.zeros_like(bone_mask, dtype=bool)
  tibia_mask = np.zeros_like(bone_mask, dtype=bool)

  # Femur: lower indices along axis < cut
  slicer = [slice(None)] * bone_mask.ndim
  slicer[axis] = slice(cut, None)
  tibia_mask[tuple(slicer)] = bone_mask[tuple(slicer)]

  slicer[axis] = slice(None, cut)
  femur_mask[tuple(slicer)] = bone_mask[tuple(slicer)]

  return femur_mask, tibia_mask

def clean_mask(mask, min_size=1000, closing_iter=2):
  #Remove small components and apply morphological closing.
  labels, num = ndimage.label(mask)
  counts = np.bincount(labels.ravel())
  counts[0] = 0  # ignore background

  remove = counts < min_size
  remove_mask = remove[labels]
  labels[remove_mask] = 0
  cleaned = labels > 0

  struct = ndimage.generate_binary_structure(3, 1)
  for _ in range(closing_iter):
      cleaned = ndimage.binary_closing(cleaned, structure=struct)
  return cleaned

def segment_bones_combined(ct_image, ct_data, output_path, hu_threshold_value):
  fig1, axes1 = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)

  # #applying gaussian filter to smooth the edges
  # smoothed_ct_data = ndimage.gaussian_filter(ct_data, sigma=sigma) #sigma=1.0

  bone_mask,labels, num_labels = threshold_and_label(ct_data, hu_threshold_value)
  # print("Bone_mask:", bone_mask)
  # print("Labels:", labels)
  print("Number of labels:", num_labels)

  # Visualize both
  visulaize_data(ct_data, axes1[0][0], "Actual Input")
  visulaize_data(bone_mask, axes1[0][1], "Bone Mask")
  visulaize_data(labels, axes1[0][2], "labels", cmap='tab10')

  femur_mask, tibia_mask = split_femur_tibia_by_slice(bone_mask, axis=2, cut_frac=0.5)
  visulaize_data(femur_mask, axes1[1][0], "Femur Mask")
  visulaize_data(tibia_mask, axes1[1][1], "Tibia Mask")

  cleaned_femur_mask = clean_mask(femur_mask)
  cleaned_tibia_mask = clean_mask(tibia_mask)

  combined_mask = np.logical_or(cleaned_femur_mask, cleaned_tibia_mask)

  visulaize_data(cleaned_femur_mask, axes1[2][0], "Cleaned Femur Mask")
  visulaize_data(cleaned_tibia_mask, axes1[2][1], "Cleaned Tibia Mask")
  visulaize_data(combined_mask, axes1[2][2], "Combined Mask")

  plt.show()

  combined_image = nib.Nifti1Image(combined_mask.astype(np.uint8), ct_image.affine)

  nib.save(combined_image, output_path)
  print(f"Saved combined femur + tibia mask to: {output_path}")


if __name__ == '__main__':
  input_path = "../input/3702_left_knee.nii"
  input_data, input_data_array = load_data(input_path)
  threshold_value = find_bone_threshold_by_histogram(input_data_array)
  output_path = "D:/bachelor/nammi/assignment-1/output/femur_tibia_mask.nii.gz"
  
  segment_bones_combined(input_data, input_data_array,output_path, threshold_value)