import os
from multiprocessing import Process
from pathlib import Path
from typing import Union

import uvicorn

from ultrack import MainConfig
from ultrack.api import app


def _in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def start_server(
    api_results_path: Union[Path, str, None] = None,
    ultrack_data_config: Union[MainConfig, None] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Starts the server.

    Parameters
    ----------
    api_results_path : Union[Path, str, None], optional
        Path to the API results folder. Defaults to None.

    """
    if api_results_path is not None:
        os.environ["API_RESULTS_PATH"] = str(api_results_path)
    if ultrack_data_config is not None:
        os.environ["ULTRACK_DATA_CONFIG"] = ultrack_data_config.json()

    if _in_notebook():

        def start_in_notebook():
            uvicorn.run(app.app, host=host, port=port)

        Process(target=start_in_notebook).start()
    else:
        uvicorn.run(app.app, host=host, port=port)


if __name__ == "__main__":
    start_server()
