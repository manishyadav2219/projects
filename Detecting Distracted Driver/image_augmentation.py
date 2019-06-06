'''
To apply image augmentation to any model add this code in the model
'''

#image augmentation 
#increasing 1000 images in each class
import random
import os
from skimage import io
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage.util import random_noise
from skimage.transform import rotate
import cv2
# our folder path containing some images
folder_path = 'G:\state-farm-distracted-driver-detection\imgs2\c9'
# the number of file to generate
num_files_desired = 1000

# loop on all files of the folder and build a list of files paths
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
num_generated_files = 0
image_to_transform=[]
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(images)
    # read image as an two dimensional array of pixels
    x= cv2.imread(image_path)
    image_to_transform.append(x)
    num_generated_files=num_generated_files+1
    #print(num_generated_files)
    
img=np.array(image_to_transform)

import cv2
folder_path='G:\state-farm-distracted-driver-detection\imgs2\c9'
i=0
new_file_path = '%s/augmented_image_%i.jpg' % (folder_path, num_generated_files)
print(new_file_path)
# write image to the disk
x=32000
for i in range(1001):
    new_file_path = '%s/img_%s.jpg' % (folder_path,x)
    x=x+1
    sk.io.imsave(new_file_path, img[i])