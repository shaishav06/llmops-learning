# %% IMPORTS
import typing as T
from pathlib import Path

import mlflow
from Agent_Recipies.io import services
from Agent_Recipies.pipelines import base
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from Agent_Recipies.io.vector_db import QdrantVectorDB
import dotenv
import os

logger = services.LoggerService().logger()


# # %% Function to test vectorDB
def test_faiss_vectordb():
    # TODO: Refactor and move to tests
    # Ensure the config path is relative to this script's location
    script_dir = str(Path(__file__).parent.parent.parent.parent)
    config_path = script_dir + "/confs/rag_chain_config.yaml"
    print("Config Path: ", config_path)

    # Load the chain's configuration
    model_config = mlflow.models.ModelConfig(development_config=config_path)
    retriever_config = model_config.get("retriever_config")

    # Load Vector Store
    embeddings = BedrockEmbeddings(model=retriever_config.get("embedding_model"))  # type: ignore
    vector_store_path = script_dir + retriever_config.get("vector_store_path")

    vector_store = FAISS.load_local(
        embeddings=embeddings, folder_path=vector_store_path, allow_dangerous_deserialization=True
    )

    # configure document retrieval
    retriever = vector_store.as_retriever(
        search_kwargs={"k": retriever_config.get("parameters")["k"]}
    )

    # Perform a similarity search
    query = "What is the content of the documents?"
    results = retriever.invoke(query)

    for doc in results:
        logger.debug(f"Content: {doc.page_content}, Metadata: {doc.metadata}")


# %% Job class for ingesting documents and updating the vector database
class IngestAndUpdateVectorDBJob(base.Job):  # type: ignore[misc]
    """Job to ingest documents and update the FAISS vector store.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
    """

    KIND: T.Literal["IngestAndUpdateVectorDBJob"] = "IngestAndUpdateVectorDBJob"

    embedding_model: str
    vector_store_path: str
    collection_name: str
    document_path: str

    @T.override
    def run(self) -> base.Locals:
        # Setup services
        # services
        # - logger
        logger = self.logger_service.logger()

        script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
        document_path = script_dir + self.document_path
        dotenv.load_dotenv(script_dir + "/.env")

        logger.info(f"Loading Documents from {document_path}...")

        embeddings = BedrockEmbeddings(model_id=self.embedding_model)

        vector_db = QdrantVectorDB(
            embeddings_model=embeddings,
            collection_name=self.collection_name,
            url=self.vector_store_path,
            api_key=os.getenv("QDRANT_API_KEY"),
            vector_size=1536,
        )

        logger.info(
            f"Ingesting documents and updating the {vector_db.__class__.__name__} vector store..."
        )

        vector_db.ingest_documents(document_path)

        logger.success("Documents ingested and vector store updated successfully")
        # test_vectordb()
        return locals()


if __name__ == "__main__":
    # Test the pipeline

    from Agent_Recipies import settings
    from Agent_Recipies.io import configs

    script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
    config_files = ["/rag_chain_config.yaml", "/feature_eng.yaml"]

    file_paths = [script_dir + "/confs/" + file for file in config_files]

    files = [configs.parse_file(file) for file in file_paths]

    config = configs.merge_configs([*files])  # type: ignore
    config["job"]["KIND"] = "IngestAndUpdateVectorDBJob"  # type: ignore

    object_ = configs.to_object(config)  # python object

    setting = settings.MainSettings.model_validate(object_)

    with setting.job as runner:
        runner.run()
