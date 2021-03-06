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
from . import ml
from . import txt
from . import stats
from . import stan

__all__ = ["core", "data", "manipulate", "math", "sys", "plot", "ml", "txt", "stats", "stan"]

