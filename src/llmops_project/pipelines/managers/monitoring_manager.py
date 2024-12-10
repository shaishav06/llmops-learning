import typing as T

from llmops_project.pipelines import base
from llmops_project.pipelines.monitoring.post_deploy_eval import MonitoringEvalJob
from pathlib import Path


# %% Job class for logging and registering the RAG model
class MonitoringJob(base.Job, frozen=True):  # type: ignore[misc]
    """Job to orchestrate Monitoring Workflow.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
    """

    KIND: T.Literal["MonitoringJob"] = "MonitoringJob"

    trace_experiment_name: str
    monitoring_experiment_name: str
    filter_string: T.Optional[str] = None

    @T.override
    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()

        logger.info("Starting Model Monitoring Workflow")

        MonitoringEvalJob(
            trace_experiment_name=self.trace_experiment_name,
            monitoring_experiment_name=self.monitoring_experiment_name,
            filter_string=self.filter_string,
        ).run()

        logger.success("Model Monitoring Workflow complete")

        return locals()


if __name__ == "__main__":
    from llmops_project import settings
    from llmops_project.io import configs

    script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
    config_files = ["/rag_chain_config.yaml", "/monitoring.yaml"]

    file_paths = [script_dir + "/confs/" + file for file in config_files]

    files = [configs.parse_file(file) for file in file_paths]

    config = configs.merge_configs([*files])  # type: ignore
    config["job"]["KIND"] = "MonitoringJob"  # type: ignore

    object_ = configs.to_object(config)  # python object

    setting = settings.MainSettings.model_validate(object_)

    with setting.job as runner:
        runner.run()
