import matplotlib.pyplot as plt
from utils import load_data

input_path = "../input/3702_left_knee.nii"

input_data, input_data_array = load_data(input_path)

print("Shape of the input data:", input_data_array.shape)
print("Affine of data:",input_data.affine)        # Spatial orientation
print("Header of dta:", input_data.header) 
print("Max intensity:",input_data_array.max()) 

# Finding the middle slice along the z-axis
mid_slice = input_data_array.shape[2] // 2

#creating the figure with two subplot
fig, axes = plt.subplots(1, 2, figsize=(10,5))

#vizualizing the mid slice on left
mid_image = input_data_array[:,:,mid_slice]
axes[0].imshow(mid_image, cmap='gray')
axes[0].set_title(f'Axial Slice {mid_slice}')
axes[0].axis('off')

# Loop through axial slices on right
for i in range(input_data_array.shape[2]):
    axes[1].imshow(input_data_array[:, :, i], cmap='gray')
    axes[1].set_title(f'Axial Slice {i}')
    axes[1].axis('off')
    plt.pause(0.1)
    axes[1].cla()

plt.close(fig)

