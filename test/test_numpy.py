import numpy as np

b = []
for i in range(10):
    b.append(np.random.random((2,3,4)))

print(np.stack(b,axis=0).shape)
print(np.stack(b,axis=1).shape)
