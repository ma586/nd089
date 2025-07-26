from typing import List

import numpy as np


my_list: List[int] = [1, 2, 4, 6, 7]
my_np_array = np.array(my_list)
print(my_np_array)
print(my_np_array.dtype)

print(my_np_array[0])
print(my_np_array[-1])

my_np_array[0] = 17
print(my_np_array)

Y = np.arange(1,10).reshape(3,3)
print(Y)
Y[1,1] = 30
print(Y)


