import numpy as np
a = np.array([[0,1,0], [1,1,1],[0,1,0]])
b = np.array([[0,1,0], [1,1,1],[0,1,0]])

equal =not (a != b).all()

print(equal)