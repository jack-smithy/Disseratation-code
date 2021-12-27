import numpy as np
import matplotlib.pyplot as plt
from ising_model import Ising

s = Ising(64, live_show=True)
results = s.simulate([2.269], 100000)
