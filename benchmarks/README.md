# ultrack benchmarks

Benchmarking ultrack with Airspeed Velocity (ASV).

## Setup

Install `asv` inside your conda environment:
```
pip install asv
```

Add your machine configuration:
```
asv machine
```

Setup the benchmark environment:
```
asv setup
```

## Usage

To run the ASV benchmark from ultrack root directory.

```
cd benchmarks
asv run
asv publish
asv preview

```

To test uncommited changes replace `run` with `dev`.


## Profiling

ASV allows profiling the `run` command by using the `--profile` flag.

Once the benchmarks were executed in profiling mode you can export the results with the `get_profile_results.py` script.
It prints the profiling results and saves it using the `pstat` format.

For example:

```
python get_profile_results.py .asv/results/<machine.name>/<run.name>.json -b "benchmark_solver.SolverSuite.time_optimize"
```

Additional information can be found on their [official documentation](https://asv.readthedocs.io/en/stable/).
