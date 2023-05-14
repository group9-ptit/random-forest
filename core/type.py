from typing import Any, Dict, TypeVar, Union, List, Set, Optional, Tuple, Literal

Record = TypeVar("Record", bound=Dict[str, float])
Label = TypeVar("Label", bound=Union[str, int])
CsvRow = TypeVar("CsvRow", bound=Dict[str, Any])
CsvRowWithoutLabel = TypeVar("CsvRowWithoutLabel", bound=Dict[str, Any])
Measure = Literal["entropy", "gini"]
