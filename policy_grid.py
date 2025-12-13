from typing import List


def load_policy_grid(path: str) -> List[List[float]]:
    """
    Load a coordinate-wise policy score grid from a text/CSV file.
    Each line should contain comma- or space-separated numeric values.
    Shape must match the map dimensions (rows x cols).
    """
    grid: List[List[float]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # support comma or space separated
            if "," in line:
                parts = line.split(",")
            else:
                parts = line.split()
            row = [float(x) for x in parts]
            grid.append(row)
    return grid

