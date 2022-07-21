import numpy as np
from scipy.spatial.transform import Rotation

from projector import Projector


def test_projector():
    orientation = Rotation.from_quat([0.0, 0.707, 0.0, 0.707]).as_euler('zyx')
    translation = np.array([-0.010591582727387028, 0.07967922530890392, 1.364566919700979])
    translation *= 1000.0
    projector = Projector()
    image_path = projector.project(image_id=0, orientation=orientation, translation=translation)
    assert len(image_path) > 0
