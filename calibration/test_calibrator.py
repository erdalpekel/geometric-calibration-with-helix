import numpy as np
from scipy.spatial.transform import Rotation
import calibration
from calibration.constants import IMAGE_WIDTH, SPHERES

from projector import Projector
from segmentation import Segmentation
from calibrator import Calibrator


def test_segmentation():
    pose = {
        "orientation": {
            "w": -0.14462377453132158,
            "x": -0.13096711608251682,
            "y": 0.7281199785576246,
            "z": -0.6570942665786899
        },
        "position": {
            "x": 0.0035938363166286846,
            "y": 0.07972184180416286,
            "z": 1.3735339204244181
        }
    }
    quat_tmp = pose["orientation"]
    pos_tmp = pose["position"]
    orientation = Rotation.from_quat([quat_tmp["x"], quat_tmp["y"], quat_tmp["z"], quat_tmp["w"]]).as_euler('zyx')
    translation = np.array([pos_tmp["x"], pos_tmp["y"], pos_tmp["z"]])
    translation *= 1000.0

    projector = Projector()
    image_path = projector.project(image_id=0, orientation=orientation, translation=translation)
    assert len(image_path) > 0

    # now segment the circles
    segmentation = Segmentation(image_path=image_path)
    circles = segmentation.segment_circles()
    expected_number_of_circles = int(len(SPHERES) / 2.0)
    assert len(circles) >= expected_number_of_circles

    calibrator = Calibrator()
    calibrator.execute_cb(image_id=0, orientation=orientation, translation=translation, sphere_correspondences=circles)
