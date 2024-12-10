"""Check tasks for pyinvoke."""

# %% IMPORTS

from invoke.context import Context
from invoke.tasks import task

# %% TASKS


@task
def poetry(ctx: Context) -> None:
    """Check poetry config files."""
    ctx.run("poetry check --lock", pty=True)


@task
def format(ctx: Context) -> None:
    """Check the formats with ruff."""
    ctx.run("poetry run ruff format --check src/ tasks/ tests/", pty=True)


@task
def type(ctx: Context) -> None:
    """Check the types with mypy."""
    ctx.run("poetry run mypy src/ tasks/ tests/", pty=True)


@task
def code(ctx: Context) -> None:
    """Check the codes with ruff."""
    ctx.run("poetry run ruff check src/ tasks/ tests/", pty=True)


@task
def test(ctx: Context) -> None:
    """Check the tests with pytest."""
    ctx.run(
        "poetry run pytest "
        "tests/pipelines/feature_engineering/test_create_vector_db.py "  # Feature Engineering
        "tests/pipelines/feature_engineering/test_ingest_documents.py "  # Feature Engineering
        "tests/pipelines/monitoring/test_generate_rag_dataset.py "  # Monitoring
        "tests/pipelines/deployment/test_register_model.py "  # Deployment
        "tests/pipelines/monitoring/test_pre_deploy_eval.py "  # Monitoring
        "tests/pipelines/deployment/test_deploy_model.py "  # Deployment
        "tests/io/test_services.py "  # IO
        "tests/io/test_configs.py "  # IO
        "tests/pipelines/test_base.py ",  # Base
        # "--numprocesses='auto'"
        pty=True,
    )


@task
def security(ctx: Context) -> None:
    """Check the security with bandit."""
    ctx.run("poetry run bandit --recursive --configfile=pyproject.toml src/", pty=True)


@task
def coverage(ctx: Context) -> None:
    """Check the coverage with coverage."""
    ctx.run(
        "poetry run pytest --cov=src/ --cov-fail-under=20 "
        "tests/pipelines/feature_engineering/test_create_vector_db.py "  # Feature Engineering
        "tests/pipelines/feature_engineering/test_ingest_documents.py "  # Feature Engineering
        "tests/pipelines/monitoring/test_generate_rag_dataset.py "  # Monitoring
        "tests/pipelines/deployment/test_register_model.py "  # Deployment
        "tests/pipelines/monitoring/test_pre_deploy_eval.py "  # Monitoring
        "tests/pipelines/deployment/test_deploy_model.py "  # Deployment
        "tests/io/test_services.py "  # IO
        "tests/io/test_configs.py "  # IO
        "tests/pipelines/test_base.py ",  # Base
        # "--numprocesses='auto'"
        pty=True,
    )


@task(pre=[poetry, format, type, code, security, coverage], default=True)
def all(_: Context) -> None:
    """Run all check tasks."""
