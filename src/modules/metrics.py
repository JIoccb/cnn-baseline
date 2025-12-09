from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError

def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection({
        'mae': MeanAbsoluteError(**kwargs),
        'mape': MeanAbsolutePercentageError(**kwargs),
        'mse': MeanSquaredError(**kwargs)
    })
