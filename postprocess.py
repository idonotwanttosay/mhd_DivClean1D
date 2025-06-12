import glob
import re
from typing import Tuple, List
import pandas as pd
import numpy as np


def available_steps(prefix: str) -> List[int]:
    """Return sorted step numbers for given prefix."""
    files = glob.glob(f"Result/out_{prefix}_*.csv")
    return sorted(int(re.findall(rf"_{prefix}_(\d+)", f)[0]) for f in files)


def read_grid(prefix: str = "rho") -> Tuple[np.ndarray, np.ndarray]:
    """Read grid coordinates from the first output file."""
    f = glob.glob(f"Result/out_{prefix}_*.csv")[0]
    df = pd.read_csv(f, header=None)
    xs = np.unique(df[0])
    ys = np.unique(df[1])
    return xs, ys


def load_field(prefix: str, step: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Load a field and reshape according to the grid."""
    df = pd.read_csv(f"Result/out_{prefix}_{step}.csv", header=None)
    return df[2].values.reshape(len(xs), len(ys)).T

