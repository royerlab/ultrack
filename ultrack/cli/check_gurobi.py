import click
from rich import print


@click.command("check_gurobi")
def check_gurobi_cli() -> None:

    try:
        import gurobipy as gp

        print("Gurobi module imported successfully.")

        # Attempt to create a Gurobi environment. This will check the license.
        gp.Env()
        print("Gurobi environment created successfully.")

    except ImportError:
        print(
            "Gurobi module not found. Ensure it is installed and the Python path is set correctly."
        )
        print("Using `conda install gurobi -c gurobi` is recommended.")
    except gp.GurobiError as e:
        print(f"GurobiError encountered: {e.message}")
        print("There may be a problem with your Gurobi license or setup.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
