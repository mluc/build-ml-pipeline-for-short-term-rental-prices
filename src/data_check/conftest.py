import uuid
import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def data(request):
    run_name = f"data_tests_data_{uuid.uuid4().hex[:8]}"
    run = wandb.init(job_type="data_tests", name=run_name, resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run_name = f"conftest_data_tests_ref_{uuid.uuid4().hex[:8]}"
    run = wandb.init(job_type="data_tests", name=run_name, resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    run_name = f"conftest_data_tests_kl_threshold_{uuid.uuid4().hex[:8]}"
    run = wandb.init(job_type="data_tests", name=run_name, resume=True)
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)

@pytest.fixture(scope='session')
def min_price(request):
    run_name = f"conftest_data_tests_min_price_{uuid.uuid4().hex[:8]}"
    run = wandb.init(job_type="data_tests", name=run_name, resume=True)
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)

@pytest.fixture(scope='session')
def max_price(request):
    run_name = f"conftest_data_tests_max_price_{uuid.uuid4().hex[:8]}"
    run = wandb.init(job_type="data_tests", name=run_name, resume=True)
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
