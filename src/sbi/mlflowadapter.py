import mlflow


class MLflowAdapter:
    log_dir = None

    def __init__(self, mlflow):
        self._mlflow = mlflow

    def log_metric(self, name, value, step=None):
        self._mlflow.log_metric(name, value, step=step)

    def log_metrics(self, metrics, step=None):
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def log_params(self, params):
        self._mlflow.log_params(params)

    def add_figure(self, name, figure, step=None):
        self._mlflow.log_figure(figure, f"{name}.png")

    def flush(self):
        pass
