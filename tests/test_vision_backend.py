import unittest

import numpy as np

import eye_state
from eye_state.vision import create_detector, infer


class VisionBackendTests(unittest.TestCase):
    def test_release_version(self):
        self.assertEqual(eye_state.__version__, "0.1.0")

    def test_native_blank_inference(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with create_detector(video=False) as detector:
            result = infer(detector, frame)

        self.assertEqual(result.face_landmarks, [])


if __name__ == "__main__":
    unittest.main()
