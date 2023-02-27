import numpy as np
class baseAM(object):
    def __init__(self):
        self.explainer_ = None
        self.local_feature_importance = None

    def set_explainer(self, ML_model, background = None):
        """
        ML_model: is an object of type baseML model
        """
        pass

    def explain(self, x):
        """
        x is a numpy array
        """
        pass

    def append_local_importance(self, importance):
        if self.local_feature_importance:
            self.local_feature_importance = np.append(self.local_feature_importance, importance, axis = 0)
        else:
            self.local_feature_importance = importance
