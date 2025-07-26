import pandas as pd
import numpy as np

fruits = pd.Series(data=[20, 40, 7], index=['apples', 'bananas', 'pineapples'])
print(fruits.index)
print(fruits.values)


print(fruits * 2)
print(fruits)



print(fruits['bananas'] + 2)