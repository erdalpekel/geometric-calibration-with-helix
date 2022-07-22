import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import os
from math import sin, cos
import logging

from constants import BASE_PATH, INTRINSIC_MATRIX, SPHERES, HELIX_PHI, HELIX_OMEGA, HELIX_RADIUS, HELIX_HEIGHT, HELIX_SAMPLING_RATE, COST_THRESHOLD_PER_SPHERE
from optimizer import Optimizer


class Calibrator:
    def __init__(self):
        self.num_spheres = len(SPHERES)

    def plot_detected_spheres(self, image_id, correspondeces, counter_correspondences, output_image):
        for i in range(0, counter_correspondences):
            index = i * 5
            sphere_correspondence = np.array([correspondeces[index + 3], correspondeces[index + 4]])

            try:
                output_image = cv2.drawMarker(output_image, tuple(
                    sphere_correspondence.astype(int)), (0, 0, 255), cv2.MARKER_CROSS, markerSize=25, thickness=2)
            except Exception as exception:
                logging.error(exception)
                logging.error("An exception occurred with image %i sphere %i" % (image_id, i))

    def calulateMatrices(self, x0, K):
        a, b, c, x, y, z = x0
        logging.debug("a, b, c, x, y, z: %f, %f, %f, %f, %f, %f", a, b, c, x, y, z)

        R = Rotation.from_euler('zyx', [a, b, c]).as_matrix()

        t = np.array([[x], [y], [z]])
        T = np.hstack((R, t))
        P = np.matmul(K, T)

        translation = np.array([x, y, z])
        camera_center = np.dot(-R.T, translation)

        return (R, translation.reshape((3,)), camera_center.reshape((3,)), P)

    def plot_helix(self, P, helix_t_range, r_ref, omega_ref, phi_ref, N_linspace, image_id, output_image_path, output_image):
        t_linspace = np.linspace(helix_t_range[0], helix_t_range[1], N_linspace)
        for i in range(0, N_linspace):
            helix_x = r_ref * cos(omega_ref * t_linspace[i] + phi_ref)
            helix_y = r_ref * sin(omega_ref * t_linspace[i] + phi_ref)
            point_tmp = np.array([helix_x, helix_y, t_linspace[i]])
            sphere_position_3d = np.array([point_tmp[0], point_tmp[1], point_tmp[2], 1])

            try:
                projection = P.dot(sphere_position_3d)
                sphere_position_2d = np.array(
                    [projection[0] / projection[2], projection[1] / projection[2]])
                output_image = cv2.drawMarker(output_image, tuple(
                    sphere_position_2d.astype(int)), (255, 0, 0), cv2.MARKER_CROSS, markerSize=15, thickness=2)
            except Exception as exception:
                logging.error(exception)
                logging.error("An exception occurred with image %i sphere %i", image_id, i)

        success_write = cv2.imwrite(output_image_path, output_image)
        logging.info("success write to %s : %r", output_image_path, success_write)

    def execute_cb(self, image_id, orientation, translation, sphere_correspondences):
        subfolder = str(HELIX_SAMPLING_RATE)
        projected_images_path = os.path.join(BASE_PATH, "projected")
        output_images_path = os.path.join(BASE_PATH, "calibrated", subfolder)
        output_guess_images_path = os.path.join(BASE_PATH, "calibration-guess")

        try:
            if not os.path.exists(output_images_path):
                os.makedirs(output_images_path)
            if not os.path.exists(output_guess_images_path):
                os.makedirs(output_guess_images_path)
        except Exception as exception:
            logging.error(exception)

        image_filename = str(image_id) + ".png"
        image_path = os.path.join(projected_images_path, image_filename)
        output_image_path = os.path.join(output_images_path, image_filename)
        output_guess_image_path = os.path.join(output_guess_images_path, image_filename)

        spheres_tmp = []
        for i in range(len(SPHERES)):
            spheres_tmp.append([SPHERES[i][0], SPHERES[i][1], SPHERES[i][2]])

        spheres_np = np.array(spheres_tmp)
        helix_middle = HELIX_HEIGHT / 2.0
        spheres_np[:, 2] -= (helix_middle * 1000.0)

        r_ref, omega_ref, phi_ref = HELIX_RADIUS, HELIX_OMEGA, HELIX_PHI
        logging.info("Helix params: radius %f, omega %f, phi %f", r_ref, omega_ref, phi_ref)

        N_linspace = HELIX_SAMPLING_RATE

        K = None
        R_initial = None
        x0 = None
        if (len(INTRINSIC_MATRIX) != 9):
            logging.error("intrinsic matrix initial not valid! Exiting!")

            logging.info("calibration NOT successful.")
            return
        else:
            K = np.array(INTRINSIC_MATRIX).reshape((3, 3))

        R_initial = orientation
        t_initial = translation

        initial_solution = np.asarray(
            (R_initial[0], R_initial[1], R_initial[2], t_initial[0], t_initial[1], t_initial[2]), dtype=float)

        correspondences = ()
        counter_correspondences = 0
        for i in range(0, len(sphere_correspondences)):
            sphere_correspondence = np.array(sphere_correspondences[i])

            if not sphere_correspondence[0] == -1.0 and not sphere_correspondence[1] == -1.0:
                counter_correspondences += 1
                point_tmp = spheres_np[i]
                correspondences += (point_tmp[0], point_tmp[1], point_tmp[2],
                                    sphere_correspondence[0], sphere_correspondence[1])
        logging.info("counter correspondences: %i", counter_correspondences)

        helix_t_range = [spheres_np[0, 2], spheres_np[-1, 2]]
        optimizer_obj = Optimizer(intrinsic_matrix=K, helix_t_range=helix_t_range, helix_3d_reference_params={
            'r': r_ref, 'omega': omega_ref, 'phi': phi_ref}, correspondences=correspondences, N_linspace=N_linspace, num_spheres=counter_correspondences)

        # plot and save initial guess to filesystem
        output_image = cv2.imread(image_path)
        self.plot_detected_spheres(image_id, correspondences, counter_correspondences, output_image)
        matrices_guess = self.calulateMatrices(initial_solution, K)
        P_guess = matrices_guess[3]
        output_guess_image = output_image.copy()
        self.plot_helix(P_guess, helix_t_range, r_ref, omega_ref, phi_ref,
                        N_linspace, image_id, output_guess_image_path, output_guess_image)

        x0 = initial_solution
        try:
            (x0, cost) = optimizer_obj.optimize_error(
                initial_solution=initial_solution, correspondeces=correspondences)

            cost = optimizer_obj.f_helix(x0)
            cost_norm = np.linalg.norm(cost.reshape((counter_correspondences, 2)), axis=1)
            cost_sum = cost_norm.sum()
            logging.info("Geometric error spheres: %f", cost_sum)
        except Exception as exception:
            logging.error(exception)

        matrices_calibration = self.calulateMatrices(x0, K)
        P_calibration = matrices_calibration[3]

        # generate two images: one for initial guess and one for result
        self.plot_helix(P_calibration, helix_t_range, r_ref, omega_ref, phi_ref,
                        N_linspace, image_id, output_image_path, output_image)

        logging.info("calibration successful.")

        return cost_sum
