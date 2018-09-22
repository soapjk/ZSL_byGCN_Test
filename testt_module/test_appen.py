
import numpy as np
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 7, 5])

op7 = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
print(op7)
# 输出
# 0.929669680201
