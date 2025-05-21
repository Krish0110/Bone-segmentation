import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def randomized_mask(original_mask_path, max_expanded_mm=2.0):
  # Readding original image
  original_image = sitk.ReadImage(original_mask_path)
  bin_mask = sitk.Cast(original_image>0, sitk.sitkUInt8)

  # Distance mapping from bone to background
  dist_map = sitk.SignedMaurerDistanceMap(bin_mask,
                                          insideIsPositive=False,
                                          useImageSpacing=True)
  
  # Sampling random radius
  random_radius = np.random.uniform(0, max_expanded_mm)
  print(f"Using random expansion r={random_radius:.2f} mm")

  random_mask = dist_map <= random_radius
  random_mask = sitk.Cast(random_mask, sitk.sitkUInt8)

  return random_mask

def visualize_masks(original_mask, expanded_mask, random_mask, labeled_mask=None):
  orig_array = sitk.GetArrayFromImage(original_mask) > 0
  exp_array = sitk.GetArrayFromImage(expanded_mask) > 0
  rand_array = sitk.GetArrayFromImage(random_mask) > 0

  if labeled_mask is not None:
    label_array = sitk.GetArrayFromImage(labeled_mask)
  else:
    # Create label on the fly: 1=original, 2=random-only
    label_array = np.zeros_like(orig_array, dtype=np.uint8)
    label_array[orig_array] = 1
    label_array[(rand_array & ~orig_array & ~exp_array)] = 2
    label_array[(exp_array & ~orig_array & ~rand_array)] = 3

  z_mid = orig_array.shape[0] // 2
  y_mid = orig_array.shape[1] // 2
  x_mid = orig_array.shape[2] // 2

  # Prepare slices for each plane
  plane_slices = [
    (orig_array[z_mid], exp_array[z_mid], rand_array[z_mid], label_array[z_mid], "Axial"),
    (orig_array[:, y_mid], exp_array[:, y_mid], rand_array[:, y_mid], label_array[:, y_mid], "Coronal"),
    (orig_array[:, :, x_mid], exp_array[:, :, x_mid], rand_array[:, :, x_mid], label_array[:, :, x_mid], "Sagittal"),
  ]

  label_cmap = colors.ListedColormap([
    (0, 0, 0),     # 0 = background
    (0, 1, 0),     # 1 = original (green)
    (1, 0, 0),     # 2 = random-only (red)
    (0, 0, 1)      # 3 = 2mm-only (blue)
  ])

  fig, axes = plt.subplots(4, 3, figsize=(15, 15))

  for col, (o, e, r, l, title) in enumerate(plane_slices):
    axes[0, col].imshow(o, cmap='gray')
    axes[0, col].set_title(f"Original {title}")
    axes[0, col].axis('off')

    axes[1, col].imshow(e, cmap='gray')
    axes[1, col].set_title(f"2mm Expanded {title}")
    axes[1, col].axis('off')

    axes[2, col].imshow(r, cmap='gray')
    axes[2, col].set_title(f"Random Expanded {title}")
    axes[2, col].axis('off')

    axes[3, col].imshow(l, cmap=label_cmap)
    axes[3, col].set_title(f"Labeled {title} (1=orig, 2=random-only, 3=2mm-only)")
    axes[3, col].axis('off')

  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  original_mask_path = "../output/femur_tibia_mask.nii.gz"
  expanded_2mm_path = "../output/femur_tibia_mask_2mm_expanded.nii.gz"
  max_expanded = 2.0
  output_path = "../output/random_mask1.nii.gz"

  original_mask = sitk.ReadImage(original_mask_path)
  expanded_mask = sitk.ReadImage(expanded_2mm_path)

  random_mask1 = randomized_mask(original_mask_path, max_expanded)
  sitk.WriteImage(random_mask1, output_path)

  visualize_masks(original_mask, expanded_mask, random_mask1)
