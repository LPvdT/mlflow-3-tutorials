import json
from typing import Any


def as_json(obj: Any) -> str:  # noqa
    try:
        payload = json.dumps(obj, indent=2)
    except json.JSONDecodeError as e:
        raise ValueError(  # noqa
            f"Object {obj} cannot be serialized to JSON: {e!s}"  # noqa
        ) from e
    else:
        return payload
