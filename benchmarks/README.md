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

To test uncommited changes replace `run` with `dev`.

```
cd benchmarks
asv run
asv publish
asv preview
```

Additional information can be found on their [official documentation](https://asv.readthedocs.io/en/stable/).
