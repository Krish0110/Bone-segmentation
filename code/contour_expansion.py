import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

#function to expand mask
def expand_mask(input_mask, output_mask, expand_mm, fg_value=1):
  #loading and reading the voxel spacing
  mask = sitk.ReadImage(input_mask)
  spacing = mask.GetSpacing()
  print("Voxel spacing (mm):", spacing)

  #computing the dilation radius in voxels
  # ceil ensures we cover at least 2 mm
  radius_voxels = [int(np.ceil(expand_mm / s)) for s in spacing]  #converting the physical expansion in mm into voxel units per axis
  print("Kernel radius (voxels):", radius_voxels)

  #applying 3d dialation
  dilate = sitk.BinaryDilateImageFilter()
  dilate.SetKernelRadius(radius_voxels)
  dilate.SetForegroundValue(fg_value)

  expanded_mask = dilate.Execute(mask)

  #saving the output
  sitk.WriteImage(expanded_mask, output_mask)
  print("2 mm expanded mask saved to:", output_mask)

  return expanded_mask

#creating label for expanded and original area
def create_label_map(original_img, expanded_mask, output_label_path):
  original_array = sitk.GetArrayFromImage(original_img) > 0 #converting to 0 and 1 value
  expanded_array = sitk.GetArrayFromImage(expanded_mask) > 0 # convertimg to 0 ad 1 value

  label_array = np.zeros_like(original_array, dtype=np.uint8)
  label_array[original_array] = 1
  label_array[expanded_array & ~original_array] = 2

  # Convert back to SITK image, copy metadata
  label_img = sitk.GetImageFromArray(label_array)
  label_img.SetOrigin(original_img.GetOrigin())
  label_img.SetSpacing(original_img.GetSpacing())
  label_img.SetDirection(original_img.GetDirection())

  sitk.WriteImage(label_img, output_label_path)
  print(f"Labelled mask saved to: {output_label_path} (0=bg,1=orig,2=ring)")
  return label_img
   
def visualize_ouput(original_img,expanded_mask, labeled_expanded_mask):
  original_array = sitk.GetArrayFromImage(original_img)
  expanded_array = sitk.GetArrayFromImage(expanded_mask)
  labeled_expanded_array = sitk.GetArrayFromImage(labeled_expanded_mask)

  # Compute mid‐slice indices
  z_mid = original_array.shape[0] // 2
  y_mid = original_array.shape[1] // 2
  x_mid = original_array.shape[2] // 2

  fig, axes = plt.subplots(3, 3, figsize=(15, 12))

  plane_slices = [
    (original_array[z_mid, :, :], expanded_array[z_mid, :, :], labeled_expanded_array[z_mid, :, :], 'Axial'),
    (original_array[:, y_mid, :], expanded_array[:, y_mid, :], labeled_expanded_array[:, y_mid, :], 'Coronal'),
    (original_array[:, :, x_mid], expanded_array[:, :, x_mid], labeled_expanded_array[:, :, x_mid], 'Sagittal'),
  ]

  for col, (orig_slice, exp_slice, label_slice, title) in enumerate(plane_slices):
    # Row 0: original
    axes[0, col].imshow(orig_slice, cmap='gray')
    axes[0, col].set_title(f"Original {title}")
    axes[0, col].axis('off')
    # Row 1: expanded
    axes[1, col].imshow(exp_slice, cmap='gray')
    axes[1, col].set_title(f"Expanded {title}")
    axes[1, col].axis('off')
    # Row 2: labelled (0=bg black,1=orig blue,2=ring red)
    axes[2, col].imshow(label_slice, cmap='tab10')
    axes[2, col].set_title(f"Labelled {title}")
    axes[2, col].axis('off')

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  #parameter definition
  input_mask = "../output/femur_tibia_mask.nii.gz"
  output_mask = "../output/femur_tibia_mask_2mm_expanded.nii.gz"
  labeled_output_mask = "../output/femur_tibia_mask_2mm_expanded_labeled.nii.gz"

  #expansion value
  expand_mm = 2.0
  fg_value = 1

  # Load original image and convert to numpy
  original_img = sitk.ReadImage(input_mask)

  expanded_mask = expand_mask(input_mask, output_mask, expand_mm, fg_value)

  labeled_expanded_mask = create_label_map(original_img, expanded_mask, labeled_output_mask)

  visualize_ouput(original_img, expanded_mask, labeled_expanded_mask)





