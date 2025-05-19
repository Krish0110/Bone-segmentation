import matplotlib.pyplot as plt
import numpy as np
from utils import load_data, find_bone_threshold_by_histogram

# load the data
input_path = "../input/3702_left_knee.nii"
input_data, input_data_array = load_data(input_path)

auto_threshold = find_bone_threshold_by_histogram(input_data_array)
print(f"Estimated threshold: {auto_threshold:.1f} HU")

#plotting the histogram
hu_values = input_data_array.flatten()
hu_values = hu_values[np.isfinite(hu_values)]
hu_values = np.clip(hu_values, -1000, 3000)

plt.figure(figsize=(10, 5))
plt.hist(hu_values, bins=1000, color='steelblue', edgecolor='black')
plt.title('Hounsfield Unit (HU) Distribution')
plt.xlabel('HU Value')
plt.ylabel('Voxel Count')
plt.grid(True)

# Initial reference line
plt.axvline(150, color='red', linestyle='--', label='Initial guess (150 HU)')

# Auto-detected threshold
plt.axvline(auto_threshold, color='green', linestyle='--', label=f'Auto threshold ({auto_threshold:.1f} HU)')

plt.legend()
plt.tight_layout()
plt.show()
