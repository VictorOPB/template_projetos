import pandas as pd
import numpy as np
from scipy.optimize import minimize
import random
from modules.get_retornos_sp import get_retornos_sp
from modules.initial_weights import get_uniform_noneg

from modules.load_data import load_data
dict_data = load_data()

ret = get_retornos_sp(dict_data, 500, 100)

print(ret)