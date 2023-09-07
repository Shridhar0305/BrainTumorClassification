# The BraTS 2019 dataset is changed from a 16 bit signed integers to 32 bit floating point numbers. The dataset is preprocessed to care for factors such as patients positins in the scanner,
# and other issues that can cause differences in brightness on the MRI. These are cared for by employing SimpleITK N4 bias correction filter on all images. 

# Next steps will include: patch extraction and kernel methods for SVM training for K fold training/80 % training-20% testing. This will then be further evaluated using evaluations. 
# next steps include getting the feature vector and the labels and training the svm using kernel. 
import SimpleITK as sitk
import numpy as np
import os
from skimage.util import view_as_windows


def apply_n4_correction(input_path, output_path):
    input_image = sitk.ReadImage(input_path)
    input_image_float = sitk.Cast(input_image, sitk.sitkFloat32)  # Convert to 32-bit float
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(input_image_float)
    sitk.WriteImage(corrected_image, output_path)

def extract_patches(image, patch_size, stride):
    image_array = sitk.GetArrayFromImage(image)  # Convert SimpleITK image to NumPy array
    patches = view_as_windows(image_array, patch_size, step=stride)
    return patches

    
input_directory = "/Downloads/BrainTumorSegmentation/InputImages"
output_directory = "/Downloads/BrainTumorSegmentation/OutputImages"
patch_size = (64, 64, 64)
stride = (32, 32, 32)

for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith(".nii"):  # Adjust file extension if needed
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_directory)
            output_path = os.path.join(output_directory, relative_path)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            apply_n4_correction(input_path, output_path)
            corrected_image = sitk.ReadImage(output_path)
            patches = extract_patches(corrected_image, patch_size, stride)

            for i, patch in enumerate(patches):
                for i, patch in enumerate(patches):
                patch_output_path = os.path.join(output_directory, f"patch_{i}.nii")
                patch_image = sitk.GetImageFromArray(patch)
                sitk.WriteImage(patch_image, patch_output_path)
print("N4 Bias Field Correction and patch extraction applied to all images.")
