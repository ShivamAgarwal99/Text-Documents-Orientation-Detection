#Code for generating images dataset for angles -90,0 and 90
import cv2
import os
import math
directory = '/home/shivam/Documents/orientation_detection/Conversion_rotation/'
#input your angle you want to rotate original image
angle = -90
i=1
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        
        img = cv2.imread(filename)
        height, width = img.shape[:2]
        image_center = (width / 2, height / 2)

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

        radians = math.radians(angle)
        sin = math.sin(radians)
        cos = math.cos(radians)
        bound_w = int((height * abs(sin)) + (width * abs(cos)))
        bound_h = int((height * abs(cos)) + (width * abs(sin)))

        rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
        rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

        rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
        cv2.imwrite('/home/shivam/Documents/orientation_detection/Conversion_rotation/Test/' + '-90.' + str(i) + '.jpg', rotated_mat)
        i = i + 1
        continue
    else:
        continue
