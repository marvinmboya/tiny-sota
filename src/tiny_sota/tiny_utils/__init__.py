from .core import get_device, load_weights, load_model, GraphMutation
from types import SimpleNamespace

from .display import (
    printBlue, printCyan, printGreen,
    printNice, printOrange, printWarn
)
ColorPrint = SimpleNamespace(
    Blue = printBlue, Green = printGreen,
    Cyan = printCyan, Orange = printOrange,
    Nice = printNice, Warn = printWarn
)

