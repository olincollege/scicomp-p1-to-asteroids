import pandas as pd


def load_astdys_prop(filepath):
    """Parses fixed-width AstDys proper element files"""
    # Define column widths based on standard AstDys output
    widths = [(0, 7), (8, 18), (19, 29), (30, 40)]
    names = ["id", "a", "e", "sini"]

    df = pd.read_fwf(filepath, colspecs=widths, names=names, skiprows=1)
    return df
