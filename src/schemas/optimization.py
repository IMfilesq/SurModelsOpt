from dataclasses import dataclass
from src.schemas.protocols import TupleProtocol

@dataclass
class OptResult:
    min_protocol : TupleProtocol
    min_val : float
    search_time : float
    n_iter : int
    n_calls : int
    