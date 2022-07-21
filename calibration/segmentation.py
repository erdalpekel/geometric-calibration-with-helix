import os
from types import NoneType
import cv2
import numpy as np

from constants import BASE_PATH, SEGMENTATION_MIN_RADIUS, SEGMENTATION_DP, SEGMENTATION_MAX_RADIUS, SEGMENTATION_MIN_DIST, SEGMENTATION_PARAM_1, SEGMENTATION_PARAM_2


class Segmentation:
    def __init__(self, image_path):
        self.segmented_images_path = os.path.join(BASE_PATH, "segmented")
        try:
            if not os.path.exists(self.segmented_images_path):
                os.makedirs(self.segmented_images_path)
        except Exception as exception:
            print(exception)

        if os.path.isfile(image_path):
            print("File exists")
            self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            print(self.image.shape)
            print(self.image.dtype)
        else:
            print("File not exist")

    def segment_circles(self):
        circles_image = cv2.HoughCircles(
            self.image, cv2.HOUGH_GRADIENT, SEGMENTATION_DP, SEGMENTATION_MIN_DIST,
            param1=SEGMENTATION_PARAM_1, param2=SEGMENTATION_PARAM_2,
            minRadius=SEGMENTATION_MIN_RADIUS, maxRadius=SEGMENTATION_MAX_RADIUS
        )

        if circles_image is NoneType:
            return []

        # draw segmented circles
        circles_image = np.uint16(np.around(circles_image))
        color_img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        circles = []
        for i in circles_image[0, :]:
            cv2.circle(color_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(color_img, (i[0], i[1]), 2, (0, 0, 255), 3)
            circles.append([i[0], i[1]])

        # write image to destination path
        image_filename = str(0) + ".png"
        image_path = os.path.join(self.segmented_images_path, image_filename)
        write_status = cv2.imwrite(image_path, color_img)

        return circles
