import numpy as np
import nibabel as nib
import csv
from scipy import ndimage

def load_mask(path):
    img = nib.load(path)
    arr = img.get_fdata() > 0
    return arr.astype(bool), img.affine

def find_tibia_landmarks(mask, affine):
    # surface voxels = mask minus its 3D erosion
    struct = ndimage.generate_binary_structure(3,1)
    eroded = ndimage.binary_erosion(mask, structure=struct)
    surface = mask & ~eroded

    # get voxel indices of surface points
    voxels = np.column_stack(np.nonzero(surface))  # shape (N,3) as (z,y,x)

    # convert to world FRU (RAS etc) coords
    # nibabel’s apply_affine expects (N,3) in (i,j,k) → (x,y,z)
    ijk = voxels[:, [2,1,0]]  # reorder (z,y,x)→(i=x,j=y,k=z)
    world = nib.affines.apply_affine(affine, ijk)  # (N,3) in mm

    # split medial vs lateral by X
    x_coords = world[:,0]
    x_mid = (x_coords.min() + x_coords.max()) / 2.0
    medial_pts  = world[x_coords < x_mid]
    lateral_pts = world[x_coords >= x_mid]

    # in each group find the point with minimal Z (inferior)
    # world[:,2] is the Z coordinate; “lowest” = smallest value if origin at head
    medial_low  = medial_pts [np.argmin(medial_pts [:,2])]
    lateral_low = lateral_pts[np.argmin(lateral_pts[:,2])]

    return medial_low, lateral_low

if __name__ == "__main__":
    masks = {
      "orig": "../output/tibia_only/tibia_mask_only.nii.gz",
      "expand_2mm": "../output/tibia_only/tibia_mask_only_2mm_expanded.nii.gz",
      "expand_4mm": "../output/tibia_only/tibia_mask_only_4mm_expanded.nii.gz",
      "rand1": "../output/tibia_only/random_mask1_tibia.nii.gz",
      "rand2": "../output/tibia_only/random_mask2_tibia.nii.gz",
    }

    landmarks = {}
    for name, path in masks.items():
        mask, affine = load_mask(path)
        medial, lateral = find_tibia_landmarks(mask, affine)
        landmarks[name] = {"medial": medial, "lateral": lateral}
        print(f"\n{name}:")
        print(f"  Medial lowest  = {medial}")
        print(f"  Lateral lowest = {lateral}")

    with open("../output/tibia_only/tibia_landmarks.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["mask","landmark","x","y","z"])
        for name, pts in landmarks.items():
            w.writerow([name, "medial",  *pts["medial"]])
            w.writerow([name, "lateral", *pts["lateral"]])
