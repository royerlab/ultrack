import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import toml
from pydantic import BaseModel, root_validator, validator

LOG = logging.getLogger(__name__)


class DatabaseChoices(Enum):
    sqlite = "sqlite"
    postgresql = "postgresql"


class DataConfig(BaseModel):
    working_dir: Path = Path(".")
    database: DatabaseChoices = "sqlite"
    address: Optional[str] = None
    n_workers: int = 1

    class Config:
        validate_assignment = True
        use_enum_values = True

    @validator("working_dir")
    def validate_working_dir_writeable(cls, value: Path) -> Path:
        """Converts string to watershed hierarchy function."""

        value.mkdir(exist_ok=True)
        try:
            tmp_path = value / f".write_test_{uuid.uuid4().hex}"
            file_handle = open(tmp_path, "w")
            file_handle.close()
            tmp_path.unlink()
        except OSError:
            raise ValueError(f"Working directory {value} isn't writable.")

        return value

    @root_validator
    def validate_postgresql_parameters(cls, values: Dict) -> Dict:
        """Validates postgresql parameters"""

        if (
            values["database"] == DatabaseChoices.postgresql.value
            and "address" not in values
        ):
            raise ValueError(
                "`data.address` must be defined when `data.database` = `postgresql`."
                "For example: postgres@localhost:12345/example"
            )

        return values

    @property
    def database_path(self) -> str:
        """Returns database path given working directory and database type."""
        if self.database == DatabaseChoices.sqlite.value:
            return f"sqlite:///{self.working_dir.absolute()}/data.db"

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
        if not self.metadata_path.exists():
            return {}

        with open(self.metadata_path) as f:
            metadata = toml.load(f)

        return metadata

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        d = super().dict(*args, **kwargs)
        d["working_dir"] = str(d["working_dir"])
        return d
