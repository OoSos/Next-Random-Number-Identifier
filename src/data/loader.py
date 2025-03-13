import warnings

warnings.warn(
    "The module src.data.loader is deprecated and will be removed in a future version. "
    "Please use src.utils.data_loader instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from the new location for backward compatibility
from src.utils.data_loader import DataLoader, DataValidator

# If needed, add any specific functionality that isn't part of the new implementation