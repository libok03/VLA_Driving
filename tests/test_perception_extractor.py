from __future__ import annotations

import numpy as np

from vla_driving.perception import PerceptionExtractor


def test_perception_extractor_shape_without_yolo() -> None:
    extractor = PerceptionExtractor({"yolo_enabled": False}, dim=32)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    features = extractor.extract(image)
    assert features.shape == (32,)
    assert features.dtype == np.float32
