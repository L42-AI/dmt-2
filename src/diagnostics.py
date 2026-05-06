import pandas as pd
import torch
import numpy as np
from functools import singledispatch

# Globals
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# === Helpers ===
## Color strings
def _green_str(s: str) -> str:
    return f"{GREEN}{s}{RESET}"
def _red_str(s: str) -> str:
    return f"{RED}{s}{RESET}"
def _yellow_str(s: str) -> str:
    return f"{YELLOW}{s}{RESET}"
def _cyan_str(s: str) -> str:
    return f"{CYAN}{s}{RESET}"

## Output header and footer
def _print_begin(title: str) -> int:
    """ Helper: Provide header of diagnosis window """

    title_str = f"+++ {title} +++"
    title_str_col = _green_str(title_str)
    print(title_str_col)
    return len(title_str)

def _print_end(title_length: int) -> None:
    """ Helper: Provide footer of diagnosis window + newline """

    print(f"{_green_str('+' * title_length)}")
    print('\n')

## Output 
def _diagnostic_output(diagnostics: dict, label: str) -> None:

    # Output header
    title_length = _print_begin(f"Data Diagnostics")

    # Print general diagnostic information
    print(f"Diagnostics for {_red_str(label)}:") 
    print(f"{diagnostics["data_type"]} {_yellow_str('| ' + str(diagnostics["val_type"])) if 'val_type' in locals() else ''}")
    print(f"Shape: {diagnostics["shape"]}")
    print(f"Number of Missing Values: {diagnostics["num_missing"]}")
    print(f"Percentage of Missing Values: {diagnostics["percent_missing"]:.2f}%")

    # Output footer
    _print_end(title_length)

## Type-dependent diagnostic information
@singledispatch
def _diagnostic_info(data) -> dict:
    raise NotImplementedError(f"No diagnosis available for {type(data)}")

@_diagnostic_info.register
def _(data: pd.DataFrame) -> dict:
    diagnostics = {
        'data_type'         : "pandas DataFrame",
        'val_type'          : ", ".join([str(vt) for vt in data.dtypes.unique() if vt != 'object']),
        'shape'             : data.shape,
        'num_missing'       : data.isna().sum().sum(),
        'percent_missing'   : (data.isna().sum().sum() / data.size) * 100 if data.size > 0 else 0
    } 
    return diagnostics

@_diagnostic_info.register
def _(data: torch.Tensor) -> dict:
    diagnostics = {
        'data_type'         : "torch Tensor",
        'val_type'          : data.dtype,
        'shape'             : tuple(data.shape),
        'num_missing'       : torch.isnan(data).sum().item(),
        'percent_missing'   : (torch.isnan(data).sum().item() / data.numel()) * 100 if data.numel() > 0 else 0
    } 
    return diagnostics

@_diagnostic_info.register
def _(data: np.ndarray) -> dict:
    diagnostics = {
        'data_type'         : "numpy ndarray",
        'val_type'          : data.dtype,
        'shape'             : data.shape,
        'num_missing'       : np.isnan(data).sum(),
        'percent_missing'   : (np.isnan(data).sum() / data.size) * 100 if data.size > 0 else 0
    } 

    return diagnostics

def diagnose_data(
    data: pd.DataFrame | np.ndarray | torch.Tensor,
    name: str,
    get : bool = False
    ) -> None | dict:
    diagnostics = _diagnostic_info(data)
    _diagnostic_output(diagnostics, name)
    if get:
        return diagnostics
    

if __name__ == "__main__":
    x1 = torch.Tensor([1, 2])
    x2 = pd.DataFrame([1, 2])
    x3 = np.array([1, 2])

    diagnose_data(x1, 'x1')
    diagnose_data(x2, 'x2')
    diagnose_data(x3, 'x3')
