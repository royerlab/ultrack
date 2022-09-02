import pstats
from pathlib import Path

import click
from asv.results import Results


@click.command()
@click.argument("result_path", type=click.Path(path_type=Path, exists=True))
@click.option("--benchmark-name", "-b", type=str, required=True)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(path_type=Path),
    default="profile.out",
    show_default=True,
)
def main(result_path: Path, benchmark_name: str, output_file: Path) -> None:
    """Helper command to visualize ASV profiling."""

    results: Results = Results.load(str(result_path))
    if benchmark_name not in results._profiles:
        raise ValueError(
            f"Could not find {benchmark_name} in profiles. Expected {results._profiles.keys()}"
        )

    profile_data_bytes = results.get_profile(benchmark_name)
    with open(output_file, "wb") as f:
        f.write(profile_data_bytes)

    p = pstats.Stats(str(output_file))
    p.sort_stats(pstats.SortKey.TIME)
    p.print_stats()


if __name__ == "__main__":
    main()
