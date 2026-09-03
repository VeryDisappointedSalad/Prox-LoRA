import difflib
from collections.abc import Collection


def raise_if_not_contained(item: str, collection: Collection[str], name: str = "Item") -> None:
    """Check that item is in collection, and if not, raise KeyError with suggestion."""
    if item not in collection:
        closest = difflib.get_close_matches(item, list(collection), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise KeyError(f"{name} not found: {item}.{closest_str}")


def format_float(x: float) -> str:
    """Format 0.03 as 3e-2 and 0.035 as 3.5e-2."""
    if x == 0:
        return "0"
    exponent = int(f"{x:e}".split("e")[-1])
    mantissa = x / (10**exponent)
    mantissa = round(mantissa * 10**3) / 10**3  # Round to ≤3 decimal places.
    if abs(mantissa - int(mantissa)) < 1e-3:
        mantissa = int(mantissa)
    return f"{mantissa}e{exponent}"
