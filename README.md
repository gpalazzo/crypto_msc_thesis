This project's objective is to predict the directional price movement of cryptocurrencies through Machine Learning (ML) models.

The project currently has 3 working and tested ML models: XGBoost, LSTM and Logistic Regression.

There are many parameters to be set in `conf/base/parameters` directory, and I really encourage people to test different combinations of them, but the ones in the production version are already tested and working.

## Environment setup
Using your preferrable environment manager, follow the steps below to install dependencies:
1. create a virtual environment
    - conda example: `conda create -n <name> python=3.8 -y && conda activate <name>`
2. install `requirements.txt`
    - `pip install -r requirements.txt`
3. install dependencies in `pyproject.toml` with `poetry`
    - `poetry lock && poetry install`

