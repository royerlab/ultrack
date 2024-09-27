import atexit
import enum
import logging
import tempfile
from datetime import datetime
from typing import Optional

import sqlalchemy as sqla
from pydantic.v1 import BaseModel, Json, validator
from sqlalchemy import JSON, Column, Enum, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

from ultrack import MainConfig
from ultrack.api.settings import settings
from ultrack.core.database import clear_all_data

LOG = logging.getLogger(__name__)

Base = declarative_base()


def _clean_db_on_exit():
    try:
        Session._temp_dir.cleanup()
    except:
        pass


class Session:
    """Singleton class to handle the database session.

    Attributes
    ----------
    _instance : sessionmaker
        The instance of the sessionmaker.
    _temp_dir : tempfile.TemporaryDirectory
        The temporary directory to store the database.
    """

    _instance: sessionmaker = None
    _temp_dir: tempfile.TemporaryDirectory = None

    def __new__(cls):
        if cls._instance is None:
            cls._temp_dir = tempfile.TemporaryDirectory()
            settings.ultrack_data_config.working_dir = cls._temp_dir.name
            atexit.register(_clean_db_on_exit)
            engine = sqla.create_engine(
                settings.ultrack_data_config.database_path, hide_parameters=False
            )
            Base.metadata.create_all(engine)
            cls._instance = sessionmaker(bind=engine)
        return cls._instance()


class ExperimentStatus(str, enum.Enum):
    """Experiment status.

    Attributes
    ----------
    NOT_PERSISTED: str = "not_persisted"
        The experiment is not persisted in the database.
    QUEUED: str = "queued"
        The experiment is queued for execution.
    INITIALIZING: str = "initializing"
        The experiment is now initializing and preprocessing data.
    DATA_LOADED: str = "data_loaded"
        The data is loaded and ready for tracking.
    SEGMENTING: str = "segmenting"
        Ultrack is segmenting the data.
    LINKING: str = "linking"
        Ultrack is linking the segments.
    SOLVING: str = "solving"
        Ultrack is optimizing the tracks.
    EXPORTING: str = "exporting"
        Ultrack is exporting the results.
    SUCCESS: str = "success"
        The experiment was successfully executed until the EXPORTING phase.
    FAILED: str = "failed"
        The experiment failed at some point.
    """

    NOT_PERSISTED = "not_persisted"
    QUEUED = "queued"
    INITIALIZING = "initializing"
    DATA_LOADED = "data_loaded"
    SEGMENTING = "segmenting"
    LINKING = "linking"
    SOLVING = "solving"
    EXPORTING = "exporting"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


class Experiment(BaseModel):
    """Experiment model.

    Attributes
    ----------
    id : Optional[int]
        The experiment id. Defaults to None.
    status : ExperimentStatus
        The experiment status. Defaults to ExperimentStatus.NOT_PERSISTED.
    name : str
        The experiment name. Defaults to "Untitled Experiment".
    start_time : Optional[datetime]
        The experiment start time. Defaults to the datetime where it was created.
    end_time : Optional[datetime]
        The experiment end time. Defaults to None and is set when the experiment
        finishes.
    std_log : str
        The captured log from the standard output. It is initially empty.
    err_log : str
        The captured log from the standard error. It is initially empty.
    config : dict
        The experiment configuration from ultrack.config.MainConfig.
    data_url : str
        The URL to the ome-zarr data.
    image_channel_or_path : Optional[str]
        The name of the image channel. Defaults to None.
    edges_channel_or_path : Optional[str]
        The name of the edges channel. Defaults to None.
    detection_channel_or_path : Optional[str]
        The name of the detection channel. Defaults to None.
    labels_channel_or_path : Optional[str]
        The name of the labels channel. Defaults to None.
    final_segments_url : Optional[str]
        The URL to the final segments. It is set when the experiment finishes and
        the segments are exported.
    tracks : Optional[Json]
        The tracks of the experiment. It is set when the experiment finishes and
        the tracks are exported.
    See Also
    --------
    ultrack.config.MainConfig: configuration model.
    """

    id: Optional[int] = None
    status: ExperimentStatus = ExperimentStatus.NOT_PERSISTED
    name: str = "Untitled Experiment"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    std_log: str = ""
    err_log: str = ""
    config: dict
    data_url: Optional[str] = None
    image_channel_or_path: Optional[str] = None
    edges_channel_or_path: Optional[str] = None
    detection_channel_or_path: Optional[str] = None
    labels_channel_or_path: Optional[str] = None
    final_segments_url: Optional[str] = None
    tracks: Optional[Json] = None

    def get_config(self) -> MainConfig:
        config = MainConfig.parse_obj(self.config)
        config.data_config = settings.ultrack_data_config
        return config

    @validator("status", pre=True, always=True)
    def check_if_id_is_valid(cls, v, values, **kwargs):
        if (
            v == ExperimentStatus.NOT_PERSISTED
            and "id" in values
            and values["id"] is not None
        ):
            raise ValueError(
                "The id cannot be set if the experiment was never "
                "persisted in the database."
            )
        elif v != ExperimentStatus.NOT_PERSISTED and "id" not in values:
            raise ValueError(
                "The id must be set if the experiment was persisted in the database."
            )
        return v

    @validator("start_time", pre=True, always=True)
    def set_start_time(cls, v):
        return v or datetime.now().isoformat()

    @validator("end_time", always=True)
    def set_end_time(cls, v, values):
        if values["status"] in [ExperimentStatus.SUCCESS, ExperimentStatus.FAILED]:
            return v or datetime.now().isoformat()
        return None


class ExperimentDB(Base):
    __tablename__ = "experiment"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    status = Column(Enum(ExperimentStatus))
    std_log = Column(Text, nullable=True)
    err_log = Column(Text, nullable=True)
    config = Column(JSON)
    start_time = Column(String)
    end_time = Column(String, nullable=True)
    data_url = Column(String)
    image_channel_or_path = Column(String, nullable=True)
    edges_channel_or_path = Column(String, nullable=True)
    detection_channel_or_path = Column(String, nullable=True)
    labels_channel_or_path = Column(String, nullable=True)
    final_segments_url = Column(String, nullable=True)
    tracks = Column(JSON, nullable=True)


def sqlalchemy_to_pydantic(instance: ExperimentDB) -> Experiment:
    """Converts SQLAlchemy object to Pydantic object."""
    return Experiment(
        **{c.name: getattr(instance, c.name) for c in instance.__table__.columns}
    )


def pydantic_to_sqlalchemy(instance: Experiment) -> ExperimentDB:
    """Converts Pydantic object to SQLAlchemy object."""
    return ExperimentDB(**instance.__dict__)


ExperimentDB.to_pydantic = sqlalchemy_to_pydantic
Experiment.to_sqlalchemy = pydantic_to_sqlalchemy


def create_experiment_instance(experiment: Experiment) -> None:
    """Persists an experiment in the database and update its id.

    Raises
    ------
    ValueError
        If the experiment could not be persisted.

    See Also
    --------
    ultrack.api.database.Experiment: experiment model.
    ultrack.api.database.update_experiment: function to update an experiment.
    """
    try:
        clear_all_data(settings.ultrack_data_config.database_path)
        session = Session()
        experiment_db = experiment.to_sqlalchemy()
        session.add(experiment_db)
        session.commit()
        experiment.id = experiment_db.id
        session.close()
    except ValueError as e:
        raise ValueError(f"Error creating experiment: {e}")


def update_experiment(experiment: Experiment) -> None:
    """Updates an experiment in the database with the new values except for the id.

    Raises
    ------
    ValueError
        If the experiment is not found in the database.

    See Also
    --------
    ultrack.api.database.Experiment: experiment model.
    ultrack.api.database.create_experiment_instance: function to create an experiment.
    """
    session = Session()
    experiment_db = session.query(ExperimentDB).filter_by(id=experiment.id).first()
    if experiment_db is None:
        raise ValueError(f"Experiment {experiment.id} not found.")
    for key, value in experiment.dict().items():
        if key != "id":
            setattr(experiment_db, key, value)
    session.commit()
    session.close()


def get_experiment(id: int) -> Experiment:
    """Get an experiment from the database.

    Parameters
    ----------
    id : int
        The id of the experiment to get.

    Returns
    -------
    Experiment
        The experiment with the given id.

    Raises
    ------
    ValueError
        If the experiment is not found in the database.
    """
    session = Session()
    experiment_db = session.query(ExperimentDB).filter_by(id=id).first()
    if experiment_db is None:
        raise ValueError(f"Experiment {id} not found.")
    session.close()
    return experiment_db.to_pydantic()
