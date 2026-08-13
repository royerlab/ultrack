import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_pyarrow_dependency_allows_patched_version() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        dependencies = tomllib.load(file)["project"]["dependencies"]

    requirements = (Requirement(dependency) for dependency in dependencies)
    pyarrow_requirement = next(
        requirement for requirement in requirements if requirement.name == "pyarrow"
    )

    assert pyarrow_requirement.specifier.contains("23.0.1")
