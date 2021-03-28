import numpy as np

a = np.array([1,2,3], dtype='int32')
print(a)

b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)

print(a.ndim)
print(b.ndim)

print(a.shape)
print(b.shape)

print(a.dtype)

print(a.itemsize)
print(a.size)
print(a.nbytes)

a = np.array([[1,2,3,4,5,6,7], [8,9,10,11,12,13,14]])
print(a)
print(a.shape)
print(a[1, 5])
print(a[1, -2])
print(a[0, :])
print(a[:, 2])
print(a[0, 1:6:2])
print(a[0, 1:-1:2])
a[1,5] = 20
print(a)
a[:,2] = 5
print(a)
a[:,2] = [1,2]
print(a)

b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)

print(b[0, 1, 1])

print(b[:,1,:])
b[:,1,:] = [[9,9],[8,8]]
print(b[:,1,:])

print(np.zeros((2,3)))
print(np.ones((4,2,2), dtype='int32'))
print(np.full((2,2), 99))
print(np.full_like(a, 4))
print(np.random.rand(4,2,3))
print(np.random.random_sample(a.shape))
print(np.random.randint(7, size=(3,3)))
print(np.random.randint(4,7, size=(3,3)))
print(np.identity(5))
arr = np.array([[1,2,3]])
r1 = np.repeat(arr,3, axis=1)
print(r1)

output = np.ones((5,5))
print(output)

z = np.zeros((3,3))
print(z)
z[1,1] = 9
print(z)

output[1:-1, 1:-1] = z
print(output)

#Be careful when copying arrays
a = np.array([1,2,3])
b = a.copy() 
b[0] = 100
print(b)
print(a)

a = np.array([1,2,3,4])
print(a)
print(a + 2)
print(a - 2)
print(a * 2)
print(a / 2)
b = np.array([1,0,1,0])
print(a + b)
print(a**2)
print(np.sin(a))

a = np.ones((2,3))
print(a)

b= np.full((3,2), 2)
print(b)

print(np.matmul(a,b))

c = np.identity(3)
print(np.linalg.det(c))

stats = np.array([[1,2,3],[4,5,6]])
print(stats)

print(np.min(stats))
print(np.max(stats, axis=1))
print(np.sum(stats, axis=0))

before = np.array([[1,2,3,4],[5,6,7,8]])
print(before)

after = before.reshape((4,2))
print(after)

v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
print(np.vstack([v1,v2,v1,v2]))

h1 = np.ones((2,4))
h2 = np.zeros((2,2))
print(np.hstack([h1,h2]))
