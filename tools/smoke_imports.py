import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mmdet.registry import TASK_UTILS

# Ensure custom modules are imported and registered.
import dntr_custom  # noqa: F401
from dntr_custom.models.task_modules.assigners import RankingAssigner  # noqa: F401
from dntr_custom.models.task_modules.iou_calculators import BboxDistanceMetric  # noqa: F401


def main():
    ranking = TASK_UTILS.build(
        dict(
            type='RankingAssigner',
            iou_calculator=dict(type='BboxDistanceMetric', constant=12.7),
            assign_metric='nwd',
            topk=3,
        )
    )
    print('OK:', ranking.__class__.__name__)


if __name__ == '__main__':
    main()
