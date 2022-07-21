import numpy as np
from scipy import optimize
from scipy.spatial.transform import Rotation
from scipy.spatial import distance_matrix
import logging
from math import sin, cos
import sys
import time


class Optimizer:
    def __init__(self, intrinsic_matrix, helix_t_range, helix_3d_reference_params, correspondences, N_linspace=200, num_spheres=50):
        self.num_spheres = num_spheres
        self.pixel_size = 0.15
        self.helix_3d_reference_params = helix_3d_reference_params
        self.N_linspace = N_linspace
        self.K = intrinsic_matrix

        self.r_ref = self.helix_3d_reference_params['r']
        self.omega_ref = self.helix_3d_reference_params['omega']
        self.phi_ref = self.helix_3d_reference_params['phi']

        self.t_data_3d = np.linspace(helix_t_range[0], helix_t_range[1], self.N_linspace)

        self.detected_spheres = np.zeros([self.num_spheres, 2])
        for index_detected in range(0, self.num_spheres):
            index = index_detected * 5
            self.detected_spheres[index_detected] = np.array(
                [correspondences[index + 3], correspondences[index + 4]])

        self.helix_points = np.zeros([self.N_linspace, 4])
        for i in range(0, self.N_linspace):
            helix_x = self.r_ref * cos(self.omega_ref * self.t_data_3d[i] + self.phi_ref)
            helix_y = self.r_ref * sin(self.omega_ref * self.t_data_3d[i] + self.phi_ref)
            helix_point = np.array([helix_x, helix_y, self.t_data_3d[i]])

            self.helix_points[i] = np.array([helix_point[0], helix_point[1], helix_point[2], 1])

    def get_P(self, a, b, c, x, y, z):
        R = Rotation.from_euler('zyx', [a, b, c]).as_matrix()

        t = np.array([[x], [y], [z]])
        T = np.hstack((R, t))

        P_sy = np.matmul(self.K, T)

        return P_sy

    def calc_norm(self, params):
        return np.linalg.norm(params[0] - params[1])

    def f_helix(self, x_init):
        a, b, c, x, y, z = x_init.tolist()

        P_sy = self.get_P(a, b, c, x, y, z)

        projections_2d = np.matmul(P_sy, self.helix_points.T)
        projections_2d = projections_2d.T
        # normalize homogenous coordinates
        projections_2d[:, 0] /= projections_2d[:, 2]
        projections_2d[:, 1] /= projections_2d[:, 2]

        # linear calc
        norms = distance_matrix(self.detected_spheres, projections_2d[:, :2])

        result = np.empty((0))
        for index_detected in range(0, self.num_spheres):
            min_index = norms[index_detected].argmin()
            residual = self.detected_spheres[index_detected] - projections_2d[min_index, :2]
            result = np.append(result, residual.astype(float), axis=0)

        return result

    def optimize_error(self, initial_solution, correspondeces):
        # check size of correspondences
        correspondences_size = len(correspondeces)
        index_camera_params = self.num_spheres * 5
        expected_correspondences_size = index_camera_params
        if correspondences_size != expected_correspondences_size:
            logging.info("correspondeces tuple size not matching expected size: %i vs expected %i",
                         correspondences_size, expected_correspondences_size)
            return np.eye(4)

        x0 = initial_solution
        logging.info("x0: %s", np.array2string(x0, precision=2, floatmode='fixed'))

        logging.info("optimizing camera parameters")

        start_time = time.time()
        logging.info("x_init:, ", np.array2string(x0, precision=2, floatmode='fixed'))

        cost = sys.float_info.max
        result = optimize.least_squares(self.f_helix, x0, method='lm', jac='2-point')
        x0 = result.x
        cost = result.cost
        logging.info("COST: %f", cost)

        logging.info("x_result: %s", np.array2string(result.x, precision=2, floatmode='fixed'))
        self.P = self.get_P(x0[0], x0[1], x0[2], x0[3], x0[4], x0[5])
        logging.info("P:\n%s", np.array2string(self.P, precision=2, floatmode='fixed'))

        logging.info("--- %s seconds ---" % (time.time() - start_time))

        return (x0, cost)
