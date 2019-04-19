class CMLExperimentMock:
    def __init__(*args, **kwargs):
        pass

    def log_metric(self, a, b):
        pass

    def log_parameters(self, a):
        pass


try:
    from comet_ml import Experiment as CMLExperiment
except ImportError as e:
    CMLExperiment = CMLExperimentMock
