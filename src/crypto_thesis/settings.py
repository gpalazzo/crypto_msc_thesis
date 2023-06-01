# -*- coding: utf-8 -*-
"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

# Instantiated project hooks.
# from crypto_thesis.hooks import ProjectHooks
# HOOKS = (ProjectHooks(),)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import ShelveStore
# SESSION_STORE_CLASS = ShelveStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# define all possible interval minutes for Binance data collecting
# raise error if the selected interval is not within the possibilities
ALL_INTERVAL_OPTS = ["1m", "3m", "5m", "15m"]
selected_interval = "15m"
assert selected_interval in ALL_INTERVAL_OPTS, "Review selected interval for collecting Binance data"

# Class that manages how configuration is loaded.
from kedro.config import TemplatedConfigLoader

CONFIG_LOADER_CLASS = TemplatedConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
# content here will be injected in the globals.yml file and then can be replicated across yml files
# keys of `globals_dict` are expected to be found in the globals.yml, otherwise it will fail
CONFIG_LOADER_ARGS = {
    "globals_pattern": "*globals.yml",
    "globals_dict": {
        "start_date": "2017-01-01",
        "end_date": "2023-01-31",
        "binance_data_interval": selected_interval
    }
}

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
