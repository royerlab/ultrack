import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import toml
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)

LOG = logging.getLogger(__name__)


class DatabaseChoices(Enum):
    sqlite = "sqlite"
    postgresql = "postgresql"
    memory = "memory"


class DataConfig(BaseModel):
    """
    Configuration for intermediate data storage and retrieval.
    """

    n_workers: int = 1
    """Number of workers for parallel processing"""

    working_dir: Path = Path(".")
    """Working directory for auxiliary files (e.g. sqlite database, metadata)"""

    database_file_name: str = "data.db"
    """Database name, used for sqlite databases, by default: `data.db`"""

    database: DatabaseChoices = "sqlite"
    """``SPECIAL``: Database type ``sqlite`` and ``postgresql`` supported"""

    address: Optional[str] = None
    """``SPECIAL``: Postgresql database path, for example, ``postgres@localhost:12345/example``"""

    in_memory_db_id: int = 0
    """
    ``SPECIAL``: Memory database id used to identify the database in memory,
    must be altered manually if multiple instances are used
    """

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
    )

    @field_validator("working_dir")
    def validate_working_dir_writeable(cls, value: Path) -> Path:
        """Converts string to watershed hierarchy function."""
        value.mkdir(exist_ok=True, parents=True)
        try:
            tmp_path = value / f".write_test_{uuid.uuid4().hex}"
            file_handle = open(tmp_path, "w")
            file_handle.close()
            tmp_path.unlink()
        except OSError:
            raise ValueError(f"Working directory {value} isn't writable.")

        return value

    @field_serializer("working_dir")
    def serialize_working_dir(self, value: Path) -> str:
        return str(value)

    @model_validator(mode="after")
    def validate_postgresql_parameters(self) -> "DataConfig":
        """Validates postgresql parameters"""

        if self.database == DatabaseChoices.postgresql.value and self.address is None:
            raise ValueError(
                "`data.address` must be defined when `data.database` = `postgresql`."
                "For example: postgres@localhost:12345/example"
            )

        return self

    @property
    def database_path(self) -> str:
        """Returns database path given working directory and database type."""
        if self.database == DatabaseChoices.sqlite.value:
            return f"sqlite:///{self.working_dir.absolute()}/{self.database_file_name}"

        elif self.database == DatabaseChoices.memory.value:
            return f"sqlite:///file:{self.in_memory_db_id}?mode=memory&cache=shared&uri=true"

        elif self.database == DatabaseChoices.postgresql.value:
            return f"postgresql://{self.address}"

        else:
            raise NotImplementedError(
                f"Dataset type {self.database} support not implemented."
            )

    @property
    def metadata_path(self) -> Path:
        return self.working_dir / "metadata.toml"

    def metadata_add(self, data: Dict[str, Any]) -> None:
        """Adds `data` content to metadata file."""
        metadata = self.metadata
        metadata.update(data)

        LOG.info(f"Updated metadata. New content {metadata}.")

        with open(self.metadata_path, mode="w") as f:
            toml.dump(metadata, f)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns metadata as dictionary."""
        if not self.metadata_path.exists():
            return {}

        with open(self.metadata_path) as f:
            metadata = toml.load(f)

        return metadata
