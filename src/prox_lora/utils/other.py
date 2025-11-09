import difflib
from collections.abc import Collection


def raise_if_not_contained(item: str, collection: Collection[str], name: str = "Item") -> None:
    """Check that item is in collection, and if not, raise KeyError with suggestion."""
    if item not in collection:
        closest = difflib.get_close_matches(item, list(collection), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise KeyError(f"{name} not found: {item}.{closest_str}")
