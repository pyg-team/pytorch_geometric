# flake8: noqa

from .link_pred import (LinkPredPrecision, LinkPredRecall, LinkPredF1,
                        LinkPredMAP, LinkPredNDCG)

link_pred_metrics = [
    'LinkPredPrecision',
    'LinkPredRecall',
    'LinkPredF1',
    'LinkPredMAP',
    'LinkPredNDCG',
]

__all__ = link_pred_metrics
