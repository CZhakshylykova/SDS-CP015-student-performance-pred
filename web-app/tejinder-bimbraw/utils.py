from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer 
import re



columns = ['absences','higher','famrel','G1','G2']