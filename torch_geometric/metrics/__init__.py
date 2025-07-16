# flake8: noqa

from .link_pred import (LinkPredAveragePopularity, LinkPredCoverage,
                        LinkPredDiversity, LinkPredF1, LinkPredHitRatio,
                        LinkPredMAP, LinkPredMetric, LinkPredMetricCollection,
                        LinkPredMRR, LinkPredNDCG, LinkPredPersonalization,
                        LinkPredPrecision, LinkPredRecall)

link_pred_metrics = [
    'LinkPredMetric',
    'LinkPredMetricCollection',
    'LinkPredPrecision',
    'LinkPredRecall',
    'LinkPredF1',
    'LinkPredMAP',
    'LinkPredNDCG',
    'LinkPredMRR',
    'LinkPredHitRatio',
    'LinkPredCoverage',
    'LinkPredDiversity',
    'LinkPredPersonalization',
    'LinkPredAveragePopularity',
]

__all__ = link_pred_metrics
