from __future__ import annotations

import numpy as np

from vla_driving.perception import PerceptionExtractor


def test_perception_extractor_shape_without_yolo() -> None:
    extractor = PerceptionExtractor({"yolo_enabled": False}, dim=32)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    features = extractor.extract(image)
    assert features.shape == (32,)
    assert features.dtype == np.float32


def test_perception_extractor_detects_red_light_box_without_yolo() -> None:
    extractor = PerceptionExtractor({"yolo_enabled": False}, dim=32)
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    image[30:90, 50:190] = [35, 35, 35]
    yy, xx = np.ogrid[:160, :240]
    red_lamp = (xx - 82) ** 2 + (yy - 60) ** 2 <= 24**2
    image[red_lamp] = [220, 20, 20]

    features = extractor.extract(image)

    assert features[8:12].argmax() == 1
