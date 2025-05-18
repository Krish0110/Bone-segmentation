import nibabel as nib 

def load_data(path):
  #loading the data
  input_data = nib.load(path)

  #converting it into 3D numpy array (height, width, depth)
  input_data_array = input_data.get_fdata()
  return input_data, input_data_array