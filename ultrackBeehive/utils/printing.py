import pandas as pd
from rich import print
from rich.table import Table


def pretty_print_df(df: pd.DataFrame, title: str, row_name: str = "") -> None:
    """Converts dataframe to rich table and prints it."""
    table = Table(title=title)

    table.add_column(row_name)
    for c in df.columns:
        table.add_column(c)

    for name, row in df.iterrows():
        row = [str(value) for value in row]
        table.add_row(name, *row)

    print(table)
