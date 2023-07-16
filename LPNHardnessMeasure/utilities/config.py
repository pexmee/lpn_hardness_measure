import json
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = BASE_DIR / "resources" / "config.json"
LOG_FILE = BASE_DIR / "resources" / "lpn.log"
DATA_FILE = BASE_DIR / "resources" / "data.json"


class Config:
    """
    Load user-specified configuration from the file located at CONFIG_FILE.
    """

    def __init__(self) -> None:
        config = self._load_config()
        self.amount_of_secrets: int = config.get("amount_of_secrets", 0)
        dim_range: List[int] = config.get("dim_range", [0, 0])
        self.dimensions = list(range(*dim_range))
        self.error_rates: List[float] = config.get("error_rates", [])
        self.sample_amounts: List[int] = config.get("sample_amounts", [])

    def _load_config(self) -> Dict[str, Any]:
        with open(CONFIG_FILE, "r") as file:
            config = json.load(file)

        return config
