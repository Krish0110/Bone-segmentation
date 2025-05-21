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

def visualize_ouput(original_img,expanded_mask):
  original_array = sitk.GetArrayFromImage(original_img)
  expanded_array = sitk.GetArrayFromImage(expanded_mask)

  # Compute mid‐slice indices
  z_mid = original_array.shape[0] // 2
  y_mid = original_array.shape[1] // 2
  x_mid = original_array.shape[2] // 2

  fig, axes = plt.subplots(2, 3, figsize=(12, 8))

  plane_titles = ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]

  # Row 0: Original mask
  axes[0, 0].imshow(original_array[z_mid, :, :],     cmap="gray"); axes[0, 0].set_title(f"Orig {plane_titles[0]}")
  axes[0, 1].imshow(original_array[:, y_mid, :],     cmap="gray"); axes[0, 1].set_title(f"Orig {plane_titles[1]}")
  axes[0, 2].imshow(original_array[:, :, x_mid],     cmap="gray"); axes[0, 2].set_title(f"Orig {plane_titles[2]}")

  # Row 1: Expanded mask
  axes[1, 0].imshow(expanded_array[z_mid, :, :], cmap="gray"); axes[1, 0].set_title(f"Exp’d {plane_titles[0]}")
  axes[1, 1].imshow(expanded_array[:, y_mid, :], cmap="gray"); axes[1, 1].set_title(f"Exp’d {plane_titles[1]}")
  axes[1, 2].imshow(expanded_array[:, :, x_mid], cmap="gray"); axes[1, 2].set_title(f"Exp’d {plane_titles[2]}")

  # Tidy up
  for ax in axes.ravel():
      ax.axis("off")

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  #parameter definition
  input_mask = "../output/femur_tibia_mask.nii.gz"
  output_mask = "../output/femur_tibia_mask_2mm_expanded.nii.gz"

  #expansion value
  expand_mm = 2.0
  fg_value = 1

  # Load original image and convert to numpy
  original_img = sitk.ReadImage(input_mask)

  expanded_mask = expand_mask(input_mask, output_mask, expand_mm, fg_value)

  visualize_ouput(original_img, expanded_mask)





