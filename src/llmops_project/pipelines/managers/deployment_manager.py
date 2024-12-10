import typing as T
from pathlib import Path

from Agent_Recipies.pipelines import base
from Agent_Recipies.pipelines.deployment.deploy_model import DeployModelJob
from Agent_Recipies.pipelines.deployment.register_model import LogAndRegisterModelJob
from Agent_Recipies.pipelines.monitoring.pre_deploy_eval import EvaluateModelJob

AUTOMATIC_DEPLOYMENT = True


# %% Job class for logging and registering the RAG model
class DeploymentJob(base.Job):  # type: ignore[misc]
    """Job to log and register the RAG model in MLflow.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
    """

    KIND: T.Literal["DeploymentJob"] = "DeploymentJob"

    # Deployment
    registry_model_name: str
    llm_model_code_path: str
    llm_confs: str
    staging_alias: str = "champion"
    production_alias: str = "production"

    # Evaluation
    qa_dataset_path: str
    alias: str
    vector_store_path: str
    metric_tresholds: dict[str, float]

    @T.override
    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()

        logger.info("Starting Model Deployment Workflow")
        logger.info("Step: Log and Register Model")

        # Log and register the model
        with LogAndRegisterModelJob(
            registry_model_name=self.registry_model_name,
            staging_alias=self.staging_alias,
            vector_store_path=self.vector_store_path,
            llm_model_code_path=self.llm_model_code_path,
            llm_confs=self.llm_confs,
        ) as log_and_register_job:
            log_and_register_job.run()

        logger.info("Step: Evaluate Model")

        # Evaluate the model
        with EvaluateModelJob(
            registry_model_name=self.registry_model_name,
            qa_dataset_path=self.qa_dataset_path,
            alias=self.alias,
            vector_store_path=self.vector_store_path,
            metric_tresholds=self.metric_tresholds,
        ) as evaluate_job:
            evaluate_job.run()

        if not AUTOMATIC_DEPLOYMENT:
            logger.warning("Automatic Deployment is disabled")
            return locals()

        else:
            logger.info("Step: Deploy Model")

        # Deploy the model
        with DeployModelJob(
            staging_alias=self.alias,
            production_alias=self.production_alias,
            registry_model_name=self.registry_model_name,
        ) as deploy_job:
            deploy_job.run()  # Automatic Deployment

        logger.success("Model Deployment Workflow complete")

        return locals()


if __name__ == "__main__":
    from Agent_Recipies import settings
    from Agent_Recipies.io import configs

    script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
    config_files = ["/deployment.yaml", "/monitoring.yaml"]

    file_paths = [script_dir + "/confs/" + file for file in config_files]

    files = [configs.parse_file(file) for file in file_paths]

    config = configs.merge_configs([*files])  # type: ignore
    config["job"]["KIND"] = "DeploymentJob"  # type: ignore

    object_ = configs.to_object(config)  # python object

    setting = settings.MainSettings.model_validate(object_)

    with setting.job as runner:
        runner.run()
