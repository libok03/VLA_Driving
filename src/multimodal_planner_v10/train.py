"""V10 entry point using the established V9 training loop."""

from multimodal_planner_v10.data import TEMPORAL_ANCHORS_S, TemporalPlannerDataset
from multimodal_planner_v10.model import TemporalResidualPlannerV10
from multimodal_planner_v9 import train as training


def main() -> None:
    training.PlannerDataset = TemporalPlannerDataset
    training.SpatialResidualSpeedPlannerV9 = TemporalResidualPlannerV10
    training.SPATIAL_ANCHORS_M = TEMPORAL_ANCHORS_S
    training.main()


if __name__ == "__main__":
    main()
