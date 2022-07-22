import cv2
import os
from scipy.spatial.transform import Rotation
import numpy as np
import logging

from constants import SPHERE_DIAMETER, INTRINSIC_MATRIX, IMAGE_HEIGHT, IMAGE_WIDTH, SPHERES, BASE_PATH, HELIX_HEIGHT


class Projector:
    def __init__(self):
        spheres_tmp = []
        for i in range(len(SPHERES)):
            spheres_tmp.append([SPHERES[i][0], SPHERES[i][1], SPHERES[i][2]])
        self.spheres = np.array(spheres_tmp)

        helix_middle = HELIX_HEIGHT / 2.0
        self.spheres[:, 2] -= (helix_middle * 1000.0)

        self.projected_images_path = os.path.join(BASE_PATH, "projected")
        try:
            if not os.path.exists(self.projected_images_path):
                os.makedirs(self.projected_images_path)
        except Exception as exception:
            print(exception)

        self.K = np.array(INTRINSIC_MATRIX).reshape((3, 3))

    def get_projection_matrix(self, orientation, translation):
        R = Rotation.from_euler('zyx', orientation).as_matrix()
        t = translation.reshape((3, 1))
        T = np.hstack((R, t))
        P = np.matmul(self.K, T)

        return P

    def project_point(self, P, point):
        projection_2d = np.matmul(P, point.T)
        # normalize homogenous coordinates
        projection_2d[0] /= projection_2d[2]
        projection_2d[1] /= projection_2d[2]

        return projection_2d

    def project(self, image_id, orientation, translation):
        image_filename = str(image_id) + ".png"
        image_path = os.path.join(self.projected_images_path, image_filename)
        image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)  # grayscale image

        # extract external geometry
        P = self.get_projection_matrix(orientation=orientation, translation=translation)

        # determine circle diameter on image plane beforehand
        # We need a P matrix with unit orientation
        P_circle = self.get_projection_matrix(orientation=np.array([0.0, 0.0, 0.0]), translation=translation)
        point_begin = np.array([0.0, 0.0, 0.0, 1.0])
        point_end = np.array([0.0, SPHERE_DIAMETER, 0.0, 1.0])
        projection_begin = self.project_point(P=P_circle, point=point_begin)
        projection_end = self.project_point(P=P_circle, point=point_end)
        circle_diameter = np.linalg.norm(projection_begin - projection_end)
        circle_radius = int(circle_diameter / 2.0)
        print("cirlce radius: ", circle_radius)

        # project spheres onto image plane with intrinsic camera matrix
        circles = []
        for sphere in self.spheres:
            sphere_np = np.array(sphere)
            sphere_np_hom = np.append(sphere_np, 1.0)
            projection_2d = self.project_point(P=P, point=sphere_np_hom)
            circles.append(np.array(projection_2d, dtype=int))

        # draw circles onto image
        for circle in circles:
            cv2.circle(image, (circle[0], circle[1]), circle_radius, (125, 125, 125), thickness=-1)

        # write image to destination path
        write_status = cv2.imwrite(image_path, image)
        if write_status is True:
            print('Image Written')
            return image_path
        else:
            print('Error saving photo')
            return ""
