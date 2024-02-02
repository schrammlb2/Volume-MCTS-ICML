from typing import ClassVar, List, Optional, Tuple, Callable, Union, cast
import numpy as np
import torch
import torch.nn as nn
from alphazero.network.utils import (
    _map_nonlinearities,
    _process_str,
)