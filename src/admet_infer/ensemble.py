import numpy as np


def new_accumulator(shape):
    return {
        "sum": np.zeros(shape, dtype=np.float64),
        "sumsq": np.zeros(shape, dtype=np.float64),
        "count": np.zeros(shape, dtype=np.int32),
        "min": np.full(shape, np.inf, dtype=np.float64),
        "max": np.full(shape, -np.inf, dtype=np.float64),
    }


def update_accumulator(accumulator, values):
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    accumulator["sum"][mask] += values[mask]
    accumulator["sumsq"][mask] += values[mask] ** 2
    accumulator["count"][mask] += 1
    accumulator["min"][mask] = np.minimum(accumulator["min"][mask], values[mask])
    accumulator["max"][mask] = np.maximum(accumulator["max"][mask], values[mask])


def finalize_accumulator(accumulator):
    count = accumulator["count"].astype(np.float64)
    mean = np.divide(
        accumulator["sum"],
        count,
        out=np.full_like(accumulator["sum"], np.nan),
        where=count > 0,
    )
    second = np.divide(
        accumulator["sumsq"],
        count,
        out=np.full_like(accumulator["sumsq"], np.nan),
        where=count > 0,
    )
    std = np.sqrt(np.maximum(second - mean**2, 0.0))

    minimum = accumulator["min"].copy()
    maximum = accumulator["max"].copy()
    minimum[accumulator["count"] == 0] = np.nan
    maximum[accumulator["count"] == 0] = np.nan

    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "count": accumulator["count"],
    }
