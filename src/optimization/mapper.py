import numpy as np
from numpy import typing as npt

def opt_to_model(opt_protocol : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Converts an optimization format protocol to a model protocol format.

    Args:
        opt_protocol: np.ndarray (2n,) The optimization protocol as 1D array of even length , where n <= 20. First n elements are time, next n elements are dose.

    Returns: np.ndarray (20, 3) The model protocol in the format [[time, dose, time_gap], ...] of length 20. Missing time and dose values are filled with zeros.
    The time_gap is calculated as the difference between consecutive time points, with the first time_gap being equal to the first time value.
    """
    print("Check whether the last diff shouldnt be negative")

    # Ensure the input is a 1D array
    if not isinstance(opt_protocol, np.ndarray) or opt_protocol.ndim != 1:
        raise ValueError("Input opt_protocol must be a 1D numpy array.")

    # Check if the input has an even length
    if len(opt_protocol) % 2 != 0:
        raise ValueError("Input opt_protocol must have an even length.")

    # Split the optimization protocol into time and dose components
    n = len(opt_protocol) // 2
    time = opt_protocol[:n]
    dose = opt_protocol[n:]

    # Calculate time gaps between consecutive time points
    time_gap = np.diff(time, prepend=0)  # Prepend 0 to maintain the same length

    # Combine time, dose, and time_gap into a single array
    model_protocol = np.column_stack((time, dose, time_gap))

    #add zeros to the beginning of the model_protocol to make it length 20
    if model_protocol.shape[0] < 20:
        zeros_to_add = 20 - model_protocol.shape[0]
        model_protocol = np.vstack((np.zeros((zeros_to_add, 3)), model_protocol))

    return model_protocol

def tuple_to_opt(protocol: list[tuple[float, float]]) -> npt.NDArray[np.float64]:
    """
    Converts a protocol in the format [(time, dose), ...] to an optimization format protocol.

    Args:
        protocol: list of tuples (time, dose) of length 20 less than or equal to 20. Each tuple represents a time and dose pair."""

    # Ensure the input is a list of tuples
    if not isinstance(protocol, list) or not all(isinstance(t, tuple) and len(t) == 2 for t in protocol):
        raise ValueError("Input protocol must be a list of tuples (time, dose).")   

    # Split the protocol into time and dose components
    time, dose = zip(*protocol)
    time = np.array(time, dtype=np.float64)
    dose = np.array(dose, dtype=np.float64)

    # Combine time and dose into a single optimization format array
    opt_protocol = np.concatenate((time, dose))

    return opt_protocol

def tuple_to_model(protocol : list[tuple[float, float]]) -> npt.NDArray[np.float64]:
    """
    Converts a protocol in the format [(time, dose), ...] to a model protocol.

    Args:
        protocol: list of tuples (time, dose) of length less than or equal to 20. Each tuple represents a time and dose pair.

    Returns:
        np.ndarray (20, 3) The model protocol in the format [[time, dose, time_gap], ...] of length 20.

    """
        # Ensure the input is a list of tuples
    if not isinstance(protocol, list) or not all(isinstance(t, tuple) and len(t) == 2 for t in protocol):
        raise ValueError("Input protocol must be a list of tuples (time, dose).")   

    opt = tuple_to_opt(protocol)
    model = opt_to_model(opt)
    return model



    
