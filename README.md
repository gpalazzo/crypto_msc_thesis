## Overview
This project's objective is to predict the directional price movement of cryptocurrencies through Machine Learning (ML) models.
The project currently has 3 working and tested ML models: XGBoost, LSTM and Logistic Regression.

There are many parameters to be set in `conf/base/parameters` directory, and I really encourage people to test different combinations of them, but the ones in the production version are already tested and working.

The code was developed using Kedro framework, official docs: https://kedro.org and https://docs.kedro.org/en/stable/index.html

## Simplified data flow
1. pull crypto pairs from Binance in a pre-defined time interval (1 minute, 5 minutes, 15 minutes, ...)
2. define the target coin, volume bar size, amount of bars looking ahead to predict and parameter tau for labeling
3. adjust data types (float, datetime, ...)
4. for the target coin
    - filter only the target coin
    - transform prices into log returns
    - accumulate volume up to the volume bar size threshold
    - define start and end times within the volume bar
    - get target time and log return of target time using the prediction amount of bars ahead
    - create target label
5. for the features
    - exclude target from the dataset
    - transform prices into log returns
    - loop over each start and end time windows (defined in the target step) and calculate features
        - features in each window must be summarized into a single data point, so if the window has 50 rows it will become only 1 row per window
6. merge both target and features into master table and drop null data
    - it will reduce the dataset for the least recent coin. example: if we have 3 coins (A, B and C) and A started in 2017, B in 2018 and C in 2019, the dataset will start in 2019
7. build and run model, and generate model reporting
8. build portfolio P&L and generate portfolio reporting
9. compare portfolio with benchmark index

## Setup
### Dependencies
Using your preferrable environment manager, follow the steps below to install dependencies:
1. create and activate a virtual environment
    - conda example: `conda create -n <name> python=3.8 -y && conda activate <name>`
2. install `requirements.txt` located at `src/requirements.txt`
    - `pip install -r requirements.txt`
3. install dependencies in `pyproject.toml` with `poetry` (you must be located in the same directory as `pyproject.toml` file)
    - `poetry lock && poetry install`

Setup pre-commit to run on every commit
1. install the dependencies as shown above
2. run `pre-commit install`
    - **Optional**: if you already have a project before installing pre-commit, you might want to check all your files, for that run `pre-commit run --all-files`

### Credentials
To collect raw data you need Binance credentials. The project expects to have 2 environment variables named `BINANCE_API_KEY` and `BINANCE_SECRET_KEY` with Binance's api key and secret, respectively.

### Default
- pre-commit is setup to run on every commit

## Assets
- `docs/diagrams` directory contains relevant diagrams for the project
- `docs/build/html/index.html` contains an HTML page with the API documentation for all the code and modules
    - to regenerate this docs, run in your terminal `kedro build-docs` in the root directory of the project
    ![API front page example](docs/images/html_api_example.png "API front page example")
- `docs/images/kedro-pipeline.png` contains the pipeline functions' execution flow
    - to regenerate this image, run in your terminal `kedro viz` and it will open a webpage where you can download it
    ![pipeline execution flow](docs/images/kedro-pipeline.png "pipeline execution flow")

## Data
- all datasets' types and paths are defined in the catalog at `conf/base/catalog` in yml files
    - the yml key is the dataset name used by the pipelines

## Parameters
- all parameters are defined at `conf/base/parameters`
- parameters with value starting with `$` are defined in runtime when building the Kedro Session
    - example: `"${start_date}"` receives the start_date from the parameter defined at `src/crypto_thesis/settings.py`
