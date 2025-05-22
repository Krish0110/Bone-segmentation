# Knee CT Segmentation and Landmark Detection

This project performs segmentation of femur and tibia bones from 3D knee CT images (.nii format), expands the segmented contours, applies random contour adjustments, and detects key landmarks on the tibia.

---

## Project Overview

- **Input:** 3D knee CT scan in NIFTI (.nii) format
- **Segmentation Method:** Threshold-based segmentation with morphological operations
- **Contour Expansion:** Using dilation based on voxel spacing
- **Landmark Detection:** Using erosion and boundary analysis
- **Output:** Segmented masks and coordinates of medial and lateral lowest points on tibia for original and expanded masks

---

## Setup Instructions

### Requirements

- Python 3.8+
- Recommended to use a virtual environment (e.g., `venv` or `conda`)

### Required Python Packages

- nibabel
- numpy
- scipy
- matplotlib
- scikit-image

### Installation

```bash
# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install nibabel numpy scipy matplotlib scikit-image
```
### Clone the repository and navigate inside the repo
```bash
git clone https://github.com/Krish0110/Bone-segmentation.git
cd Bone-segmentation
```
###How to run
First navigate to code directory as
```bash
cd code
```
- `segmentation.py`: Performs segmentation of femur and tibia from the input `.nii` file.
  ```bash
  python segmentation.py
  ```
- `contour_expansion.py`: Expands the segmented bone contours using morphological dilation.
  ```bash
  python contour_expansion.py
  ```
- `randomized_contour_adjustment.py`: Expands the segmented bone contours using morphological dilation within the given expansion limit randomly.
  ```bash
  python randomized_contour_expansion.py
  ```
- `find_points_in_tibia.py`: Detects medial and lateral lowest points on the tibia mask.
  ```bash
  python find_points_in_tibia.py 
  ```

