import numpy as np

x = np.array([1, 2, 3, 45554444444444555, 5])

print(x)
print(type(x))

print(x.dtype)
print(x.shape)
print(x.size)
print(x.ndim)

x = np.zeros((4,4), dtype=int)
print(x)
print(x.dtype)
y = np.eye(5)

print(y)

a = np.diag([1, 2, 3])
print(a)


b = np.arange(20).reshape((10, 2))
print(b)

c = np.random.random((3,3))
print(c)

d = np.random.randint(3,5,(3,2))
print(d)


z = np.random.normal(0, 0.1, (100,100))
print(z)

print(f"mean: {z.mean()}")
print(f"std: {z.std()}")
print(f"max: {z.max()}")
print(f"min: {z.min()}")
print(f"pos: {(z > 0).sum()}")
print(f"neg: {(z < 0).sum()}")
