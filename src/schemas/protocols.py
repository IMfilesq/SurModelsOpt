from typing import TypeAlias
import numpy as np
from jaxtyping import Float64

#main/user format
TupleProtocol = list[tuple[float, float] | list[float]]

#optimizer format
FlatProtocol = Float64[np.ndarray, "N"]

#model (prediction) format
MatrixProtocol = Float64[np.ndarray, "20 3"]