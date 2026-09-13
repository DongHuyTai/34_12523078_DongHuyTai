import re
from typing import List, Union


def preprocess_features(features: Union[List[str], str]) -> List[str]:
    if isinstance(features, str):
        tokens = re.findall(r"[a-zA-Z0-9_]+", features.lower())
        cleaned = [token.strip() for token in tokens if token.strip()]
    elif isinstance(features, list):
        cleaned = []
        for item in features:
            if not isinstance(item, str):
                raise ValueError("all feature values must be strings")
            cleaned.extend(re.findall(r"[a-zA-Z0-9_]+", item.lower()))
    else:
        raise ValueError("features must be a list or a text string")

    if not cleaned:
        raise ValueError("features cannot be empty")

    return cleaned
