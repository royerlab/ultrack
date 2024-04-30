import asyncio
import io
import json
import logging
import threading
import time
import traceback as tb
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional

from fastapi import HTTPException, WebSocket
from starlette.websockets import WebSocketState

from ultrack.api.database import Experiment, ExperimentStatus

LOG = logging.getLogger(__name__)


def _mimic_carriage_return(text: str) -> str:
    """Mimic the carriage return `\r` behavior of the output text. Otherwise, text
    outputs that heavily relies on updating the line with the carriage return,
    like tqdm for example, will flood the websocket with multiple lines.
    """
    parts = text.split("\r")
    current_line = ""
    final_output = []

    for part in parts:
        if "\n" in part:
            lines = part.split("\n")
            current_line += lines[0]
            final_output.append(current_line)
            final_output.extend(lines[1:-1])
            current_line = lines[-1]
        else:
            current_line = part

    if current_line:
        final_output.append(current_line)

    return "\n".join(final_output)


class ProcessedException(HTTPException):
    """Exception that has been processed by the API."""


async def raise_error(
    e: Exception,
    exp: Optional[Experiment] = None,
    ws: Optional[WebSocket] = None,
    additional_info: Optional[str] = None,
) -> None:
    """Raise an error and send it to the websocket.

    Finishes the experiment and sends the error message to the websocket.  If the
    exception is a ProcessedException, it will not be raised again.

    Parameters
    ----------
    e : Exception
        The exception to be raised.
    exp : Optional[Experiment]
        The experiment that failed. Defaults to None if the exception is not related to
        an initialized experiment.
    ws : Optional[WebSocket]
        The websocket to send the error message. Defaults to None if the exception is
        not related to a connected websocket.
    additional_info : Optional[str]
        Additional information to be sent to the websocket. Defaults to None.

    Raises
    ------
    ProcessedException
        Wraps the original exception to indicate that it has been processed by the API
        and should not be raised again.
    """
    if not isinstance(e, ProcessedException):
        from ultrack.api.app import finish_experiment

        LOG.error(e)
        msg = ""
        if exp is not None:
            msg = f"Experiment {exp.id} failed.\n"
            exp.status = ExperimentStatus.ERROR
            await finish_experiment(ws, exp)
        if additional_info is not None:
            msg += additional_info + "\n"
        msg += f"Exception: {str(e)}\n"
        msg += "Traceback:\n" + tb.format_exc()
        if ws is not None and ws.client_state == WebSocketState.CONNECTED:
            await ws.send_text(msg)
        raise ProcessedException(status_code=500, detail=str(msg))


class UltrackWebsocketLogger:
    """Context manager to log experiment output to a websocket.

    This context manager logs the stdout and stderr of a block of code to a websocket.

    Attributes
    ----------
    websocket : WebSocket
        The websocket to send the logs.
    experiment : Experiment
        The experiment to be logged.
    ctx_stdout_switcher : redirect_stdout
        The context manager wrapped to redirect the stdout.
    ctx_stderr_switcher : redirect_stderr
        The context manager wrapped to redirect the stderr.
    stdout : io.StringIO
        The buffer to store the stdout.
    stderr : io.StringIO
        The buffer to store the stderr.
    stop_logger_event : threading.Event
        The event to stop the logger thread.

    Example
    -------
    >>> experiment = ...
    >>> websocket = ...
    >>> async with UltrackWebsocketLogger(websocket, experiment):
    ...     await start_experiment(websocket, experiment)
    ...     ...
    ...     await segment_link_and_solve(websocket, experiment)
    ...     ...
    ... await finish_experiment(websocket, experiment)
    """

    interruption_handlers: dict[int, threading.Event] = {}

    @staticmethod
    def register_interruption_handler(exp_id: int) -> threading.Event:
        """Register an interruption thread handler for the given experiment id.

        It should be used to interrupt the experiment execution thread
        from another thread.

        Parameters
        ----------
        exp_id : int
            The experiment id.

        Returns
        -------
        threading.Event
            The interruption handler.

        See Also
        --------
        UltrackWebsocketLogger.request_interruption : request the interruption of the
            experiment execution thread.
        """
        handler = threading.Event()
        UltrackWebsocketLogger.interruption_handlers[exp_id] = handler
        return handler

    @staticmethod
    def request_interruption(exp_id: int) -> None:
        """Request the interruption of the experiment execution thread.

        Parameters
        ----------
        exp_id : int
            The experiment id.

        See Also
        --------
        UltrackWebsocketLogger.register_interruption_handler : register an interruption
            handler for the given experiment id.
        """
        # handler = UltrackWebsocketLogger.interruption_handlers.get(exp_id)
        # if handler is not None:
        #     handler.set()
        # del UltrackWebsocketLogger.interruption_handlers[exp_id]
        raise NotImplementedError(
            "Interruption handler is not implemented yet due to asyncio limitations."
        )

    def __init__(
        self,
        websocket: WebSocket,
        experiment: Experiment,
        tick_rate: float = 1.0,
    ):
        """Initialize the UltrackWebsocketLogger context manager.

        Parameters
        ----------
        websocket : WebSocket
            The websocket to send the logs.
        experiment : Experiment
            The experiment to be logged.
        tick_rate : float
            The rate (in sec.) to send the logs to the websocket, by default 5.0 (sec.).
        """
        self.websocket = websocket
        self.experiment = experiment
        self.ctx_stdout_switcher = redirect_stdout(io.StringIO())
        self.ctx_stderr_switcher = redirect_stderr(io.StringIO())
        self.stdout = None
        self.stderr = None

        async def run_thread():
            """Logging thread function to be run in the background."""
            if self.stdout is not None:
                experiment.std_log = _mimic_carriage_return(self.stdout.getvalue())
            if self.stderr is not None:
                experiment.err_log = _mimic_carriage_return(self.stderr.getvalue())
            if websocket.state == WebSocketState.DISCONNECTED:
                return False
            await websocket.send_json(json.loads(experiment.json()))
            return True

        def between_callback(stop_logger_event: threading.Event):
            """Helper function to run the logging thread in the background. It is
            necessary to have it since the asyncio loop should run in a separate thread.
            """
            while not stop_logger_event.is_set():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                still_connected = loop.run_until_complete(run_thread())
                if not still_connected:
                    stop_logger_event.set()
                    raise Exception(
                        "Websocket was disconnected. "
                        "The experiment will be interrupted."
                    )
                loop.close()
                if (
                    experiment.id in UltrackWebsocketLogger.interruption_handlers
                    and UltrackWebsocketLogger.interruption_handlers[
                        experiment.id
                    ].is_set()
                ):
                    stop_logger_event.set()
                    raise Exception("Experiment was interrupted.")
                time.sleep(tick_rate)

        self.stop_logger_event = threading.Event()

        self.thread = threading.Thread(
            target=between_callback,
            args=(self.stop_logger_event,),
        )

    async def __aenter__(self) -> "UltrackWebsocketLogger":
        """Enter the context manager.

        It starts the logging thread and redirects the stdout and stderr to the
        buffers.
        """
        self.stdout = self.ctx_stdout_switcher.__enter__()
        self.stderr = self.ctx_stderr_switcher.__enter__()
        self.thread.start()

        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        """Exit the context manager.

        It stops the logging thread and redirects the stdout and stderr back to the
        original streams. It also sends the logs to the websocket if an exception was
        raised.
        """
        if exc_type is not None:
            self.stop_logger_event.set()
            self.thread.join()
            exc_obj = exc_type(exc_value).with_traceback(traceback)
            if not isinstance(exc_obj, ProcessedException):
                await raise_error(exc_obj, self.experiment, ws=self.websocket)
        self.experiment.std_log = _mimic_carriage_return(self.stdout.getvalue())
        self.experiment.err_log = _mimic_carriage_return(self.stderr.getvalue())
        self.stop_logger_event.set()
        self.thread.join()
        self.ctx_stdout_switcher.__exit__(exc_type, exc_value, traceback)
        self.ctx_stderr_switcher.__exit__(exc_type, exc_value, traceback)
        return False
