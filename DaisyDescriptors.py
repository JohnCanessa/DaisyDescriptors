# **** ****
import matplotlib.pyplot as plt

import skimage
from skimage import color
from skimage.feature import daisy
from skimage import data


# **** load the RGB pexels-cars.jpg image ****
cars = skimage.io.imread('./images/pexels-cars.jpg')

# **** show RGB cars image ****
plt.figure(figsize=(10, 8))                 # set the size of the figure
plt.imshow(cars, interpolation='nearest')   # display the image
plt.title('cars')                           # set the title
plt.show()                                  # show the image


# **** convert the car image to grayscale ****
cars = color.rgb2gray(cars)

# **** show cars image ****
plt.figure(figsize=(10, 8))                 # set the size of the figure
plt.imshow(cars, cmap='gray')               # display the image
plt.title('cars - gray')                    # set the title
plt.show()                                  # show the image


# **** compute the daisy descriptors for the cars image ****
descs_array, descs_image = daisy(   cars, 
                                    step=200,           # distance between descriptor sampling points
                                    radius=20,          # radius (in pixels) of the outermost ring
                                    visualize=True)     # generate a visualization of the DAISY descriptors

# ****  show the daisy descriptors number ****
descs_num = descs_array.shape[0] * descs_array.shape[1]

# **** show the daisy descriptors number 
#      features lie on high-contrast regions such as object edges ****
print(f'descs_num: {descs_num}')


# **** show the descs_image ****
plt.figure(figsize=(12, 10))                 # set the size of the figure
plt.title(label='descs_image')               # set the title
plt.imshow(descs_image)                      # display the image
plt.show()                                   # show the image

