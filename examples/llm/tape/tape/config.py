from enum import Enum


class DatasetName(str, Enum):
    PUBMED = 'pubmed'
    OGBN_ARXIV = 'ogbn_arxiv'


class FeatureType(str, Enum):
    TITLE_ABSTRACT = 'TA'
    PREDICTION = 'P'
    EXPLANATION = 'E'
    TAPE = 'TAPE'
