# Building docs instructions

This assumes you have already cloned the repository and are in the root directory of the repository.

Go to the docs directory and install the requirements

```bash
cd docs
pip install '..[docs]'
```

Clean and build the docs with

```bash
make clean
make html
```

In Linux, open the generated html file with

```bash
xdg-open build/html/index.html
```

or in macOS

```bash
open build/html/index.html
```
