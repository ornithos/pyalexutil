import os
if os.getcwd().split('/')[-1] == 'pyalexutil':
    raise Exception("""Working directory is within pyalexutil. Will cause path
    conflicts with standard library. Please change directory""")

from . import math
from . import sys
from . import core
from . import data
from . import manipulate
from . import plot

__all__ = ["core", "data", "manipulate", "math", "sys", "plot"]
