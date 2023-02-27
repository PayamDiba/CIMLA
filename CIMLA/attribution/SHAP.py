from CIMLA.attribution.BaseModels import baseAM
import shap
import numpy as np

class treeSHAP(baseAM):
    def __init__(self):
        super(treeSHAP,self).__init__()
        self.local_feature_importance = None

    def set_explainer(self, ML_model, background = None, model_output = "raw"):
        self.explainer_ = shap.TreeExplainer(ML_model.model, model_output = model_output)

    def explain(self, x):
        return self.explainer_.shap_values(x)

class deepSHAP(baseAM):
    def __init__(self):
        super(deepSHAP,self).__init__()
        self.local_feature_importance = None

    def set_explainer(self, ML_model, background, model_output = "raw"):
        self.explainer_ = shap.DeepExplainer(ML_model.model, background)

    def explain(self, x):
        return self.explainer_.shap_values(x)[0]
