"""High-level pipelines of the project."""

# %% IMPORTS

from Agent_Recipies.pipelines.deployment.deploy_model import DeployModelJob
from Agent_Recipies.pipelines.deployment.register_model import LogAndRegisterModelJob
from Agent_Recipies.pipelines.feature_engineering.create_vector_db import CreateVectorDBJob
from Agent_Recipies.pipelines.feature_engineering.ingest_documents import IngestAndUpdateVectorDBJob
from Agent_Recipies.pipelines.managers.deployment_manager import DeploymentJob
from Agent_Recipies.pipelines.managers.feature_engineering_manager import FeatureEngineeringJob
from Agent_Recipies.pipelines.monitoring.generate_rag_dataset import GenerateRagDatasetJob
from Agent_Recipies.pipelines.monitoring.post_deploy_eval import MonitoringEvalJob
from Agent_Recipies.pipelines.monitoring.pre_deploy_eval import EvaluateModelJob
from Agent_Recipies.pipelines.managers.monitoring_manager import MonitoringJob

# %% TYPES

JobKind = (
    DeploymentJob
    | FeatureEngineeringJob
    | GenerateRagDatasetJob
    | EvaluateModelJob
    | CreateVectorDBJob
    | IngestAndUpdateVectorDBJob
    | DeployModelJob
    | LogAndRegisterModelJob
    | MonitoringEvalJob
    | MonitoringJob
)

# %% EXPORTS

__all__ = [
    "DeploymentJob",
    "FeatureEngineeringJob",
    "GenerateRagDatasetJob",
    "EvaluateModelJob",
    "CreateVectorDBJob",
    "IngestAndUpdateVectorDBJob",
    "DeployModelJob",
    "LogAndRegisterModelJob",
    "MonitoringEvalJob",
    "MonitoringJob",
    "JobKind",
]
