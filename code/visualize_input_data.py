import matplotlib.pyplot as plt
from utils import load_data

input_path = "../input/3702_left_knee.nii"

input_data, input_data_array = load_data(input_path)

print("Shape of the input data:", input_data_array.shape)
print("Affine of data:",input_data.affine)        # Spatial orientation
print("Header of dta:", input_data.header) 
print("Max intensity:",input_data_array.max()) 

# 3. Show the mid‐slice in each plane
mid_x, mid_y, mid_z = [dim//2 for dim in input_data_array.shape]

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(input_data_array[mid_x, :, :].T, origin='lower', cmap='gray', aspect='equal')
axs[0].set_title(f'Sagittal (slice {mid_x})')

axs[1].imshow(input_data_array[:, mid_y, :].T, origin='lower', cmap='gray', aspect='equal')
axs[1].set_title(f'Coronal (slice {mid_y})')

axs[2].imshow(input_data_array[:, :, mid_z].T, origin='lower', cmap='gray', aspect='equal')
axs[2].set_title(f'Axial (slice {mid_z})')

for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()

# Loop through axial slices on right
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
for y in range(input_data_array.shape[1]):
    ax2.imshow(input_data_array[:, y, :].T, origin='lower',cmap='gray',aspect='equal')
    ax2.set_title(f'Coronal Slice {y+1}/{input_data_array.shape[1]}')
    ax2.axis('off')
    plt.pause(0.1)
    ax2.cla()

plt.close(fig2)



