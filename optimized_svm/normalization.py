import numpy as np
from sklearn.exceptions import NotFittedError


class NormalizationMixin:
    def _normalize_labels(self, y):
        if self.kind in ['classification', 'balanced_classification']:
            self.classes_ = np.unique(y)
            assert len(self.classes_) == 2
            return np.where(y == self.classes_[0], -1.0, +1.0)
        elif self.kind == 'regression':
            self.offset_ = np.median(y)
            self.scale_ = np.abs(y - self.offset_).mean()
            return (y - self.offset_) / self.scale_
        elif self.kind == 'ls':
            self.offset_ = np.mean(y)
            self.scale_ = np.sqrt(np.mean((y - self.offset_) ** 2))
            return (y - self.offset_) / self.scale_

    def _denormalize_decisions(self, d):
        if self.kind in ['classification', 'balanced_classification']:
            if not hasattr(self, 'classes_'):
                raise NotFittedError()
            return np.where(d < 0.0, self.classes_[0], self.classes_[1])
        elif self.kind in ['regression', 'ls']:
            return d * self.scale_ + self.offset_
