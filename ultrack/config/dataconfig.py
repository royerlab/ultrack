from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ValidationError, validator


class DataBaseChoices(Enum):
    sqlite = "sqlite"


class DataConfig(BaseModel):
    working_dir: Path = Path(".")
    database: DataBaseChoices = "sqlite"

    @validator("working_dir")
    def validate_working_dir_writeable(cls, value: Path) -> Path:
        """Converts string to watershed hierarchy function."""

        value.mkdir(exist_ok=True)
        try:
            tmp_path = value / ".write_test"
            file_handle = open(tmp_path, "w")
            file_handle.close()
            tmp_path.unlink()
        except OSError:
            ValidationError(f"Working directory {value} isn't writable.")

        return value

    @property
    def database_path(self) -> str:
        """Returns database path given working directory and database type."""
        if self.database == "sqlite":
            return f"sqlite:///{self.working_dir.absolute()}/data.db"
        else:
            raise NotImplementedError(
                f"Dataset type {self.database} support not implemented."
            )
