from numba import jit
import math

@jit
def hypot(x,y):
    x = abs(x);
    y = abs(y);
    t = min(x,y);
    x = max(x,y);
    t = t/x;
    return x * math.sqrt(1+t*t)

hypot(3.0,4.0)