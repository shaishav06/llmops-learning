import typing as T

from llmops_project.pipelines import base


# %% Job class for logging and registering the RAG model
class DeployModelJob(base.Job):  # type: ignore[misc]
    """Job to log and register the RAG model in MLflow.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
    """

    KIND: T.Literal["DeployModelJob"] = "DeployModelJob"

    staging_alias: str = "champion"
    production_alias: str = "production"
    registry_model_name: str

    def promote_model_to_alias(
        self, client, model_name, current_alias: str = "champion", new_alias: str = "production"
    ) -> None:
        logger = self.logger_service.logger()

        # Retrieve the model version using the current alias
        model_version = client.get_model_version_by_alias(name=model_name, alias=current_alias)

        # Access and print the tags of the model version
        if model_version.tags["passed_tests"] == "True":
            logger.success("Model version passed tests, promoting to production")
            # Set the new alias to the retrieved model version
            client.set_registered_model_alias(
                name=model_name, alias=new_alias, version=model_version.version
            )

        else:
            logger.warning("Model version did not pass tests, archiving model")
        client.delete_registered_model_alias(name=model_name, alias=current_alias)

    @T.override
    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()

        # - mlflow
        client = self.mlflow_service.client()
        logger.info("With client: {}", client.tracking_uri)

        logger.info(
            f"Deploying Model Named {self.registry_model_name} from {self.staging_alias} to {self.production_alias}"
        )
        self.promote_model_to_alias(
            client=client,
            model_name=self.registry_model_name,
            current_alias=self.staging_alias,
            new_alias=self.production_alias,
        )

        logger.success("Model deployment complete")

        return locals()


if __name__ == "__main__":
    from pathlib import Path

    from llmops_project import settings
    from llmops_project.io import configs

    script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
    config_files = ["/deployment.yaml"]

    file_paths = [script_dir + "/confs/" + file for file in config_files]

    files = [configs.parse_file(file) for file in file_paths]

    config = configs.merge_configs([*files])  # type: ignore
    config["job"]["KIND"] = "DeployModelJob"  # type: ignore

    object_ = configs.to_object(config)  # python object

    setting = settings.MainSettings.model_validate(object_)

    with setting.job as runner:
        runner.run()
