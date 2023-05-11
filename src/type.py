from typing import Any, Dict, TypeVar

Record = TypeVar('Record', bound=Dict[str, float])
CsvRow = TypeVar('CsvRow', bound=Dict[str, Any])
CsvRowWithoutLabel = TypeVar('CsvRowWithoutLabel', bound=Dict[str, Any])
