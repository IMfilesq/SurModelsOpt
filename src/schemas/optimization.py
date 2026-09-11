from dataclasses import dataclass
from src.schemas.protocols import TupleProtocol

@dataclass
class OptResult:
    opt_name : str
    start : TupleProtocol
    min_protocol : TupleProtocol
    min_val : float
    search_time : float
    n_iter : int
    n_calls : int
    