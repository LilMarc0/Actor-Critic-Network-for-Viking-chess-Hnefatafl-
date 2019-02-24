
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes = 4)
from time import time

t = time()

def pl(seq):
    for x in seq:
        if x == 15:
            return True

a = [[_ for _ in range(y)] for y in range(10000)]

for x in a:
    async_result = pool.apply_async(pl, ([x]))
# for x in a:
#     print(pl(x))
return_val = async_result.get()
print(return_val)
print(time() - t)