

from MixedPrecision.tools.loggers.cml_mock import CMLExperiment
from MixedPrecision.tools.loggers.default_logger import DefaultLogger


logger_impl = {
    'default': DefaultLogger,
    'comet_ml': CMLExperiment
}


def make_log(backend, *args, **kwargs):
    return logger_impl[backend](*args, **kwargs)
