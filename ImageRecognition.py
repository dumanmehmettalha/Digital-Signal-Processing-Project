import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d, convolve2d
from PIL import Image
import cv2
from skimage import color
from skimage import io
from scipy.signal import find_peaks

# Import the images from the Data folder and convert them to grayscale
folder_path = 'Data'
file_list = os.listdir(folder_path)


# Define your own convolution function
def convolution(image, kernel):
    m, n = image.shape
    k, l = kernel.shape
    output = np.zeros((m - k + 1, n - l + 1))
    for i in range(m - k + 1):
        for j in range(n - l + 1):
            output[i][j] = int(np.sum(image[i:i + k, j:j + l] * kernel))
    return output


def compare():
    # Import the filter image from the project folder and convert it to grayscale
    filt = io.imread('kernel.jpg', as_gray=True)
    for i in range(len(file_list)):
        # Load the image
        img = io.imread(os.path.join(folder_path, file_list[i]))
        # Convert it to grayscale
        img_gray = color.rgb2gray(img)

        # Apply the convolution using your own function
        conv1 = convolution(img_gray, filt)

        # Apply the convolution using the library function
        conv2 = convolve2d(img_gray, filt, boundary='symm')

        # Apply the correlation using the library function
        corr = correlate2d(img_gray, filt, 'full', boundary='symm')

        # Apply find_peaks function to conv1
        peaks1, _ = find_peaks(conv1.ravel(), height=2705, distance=len(conv1), width=len(filt))
        # Apply find_peaks function to conv2
        peaks2, _ = find_peaks(conv2.ravel(), height=2705,distance=len(conv2), width=len(filt))
        # Apply find_peaks function to corr
        peaks3, _ = find_peaks(corr.ravel(), height=2705,distance=len(corr), width=len(filt))

        # Print the number of peaks for each matrix
        print(f"For image {file_list[i]}:")
        print(f"- Own convolution function detected {len(peaks1)} oil tanks.")
        print(f"- Library convolution function detected {len(peaks2)} oil tanks.")
        print(f"- Correlation function detected {len(peaks3)} oil tanks.")


# Compare the results of the different methods
compare()
