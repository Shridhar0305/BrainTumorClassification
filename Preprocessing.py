# The BraTS 2019 dataset is changed from a 16 bit signed integers to 32 bit floating point numbers. The dataset is preprocessed to care for factors such as patients positins in the scanner,
# and other issues that can cause differences in brightness on the MRI. These are cared for by employing SimpleITK N4 bias correction filter on all images. 

# Next steps will include: patch extraction and kernel methods for SVM training for K fold training/80 % training-20% testing. This will then be further evaluated using evaluations. 

import SimpleITK as sitk
import numpy as np
import os


def apply_n4_correction(input_path, output_path):
    input_image = sitk.ReadImage(input_path)
    input_image_float = sitk.Cast(input_image, sitk.sitkFloat32)  # Convert to 32-bit float
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(input_image_float)
    sitk.WriteImage(corrected_image, output_path)
    
input_directory = "/kaggle/input"
output_directory = "/kaggle/working"

for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith(".nii"):  # Adjust file extension if needed
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_directory)
            output_path = os.path.join(output_directory, relative_path)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            apply_n4_correction(input_path, output_path)

print("N4 Bias Field Correction applied to all images.")
