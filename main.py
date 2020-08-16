import numpy as np
from models._MF import MF
items = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7']
num_users = 7
dict_rate = {0: [1, 0, 0, 1, 3],
             1: [2, 0, 3, 1, 1],
             2: [1, 2, 0, 5, 0],
             3: [1, 0, 0, 4, 4],
             4: [2, 1, 5, 4, 0],
             5: [5, 1, 5, 4, 0],
             6: [0, 0, 0, 1, 0], }

# P, Q is (7 X k), (k X 5) matrix
factorizer = MF(dict_rate, latent=3, lr=0.01, reg_param=0.01, epochs=300, verbose=True)
factorizer.fit()
factorizer.print_results()
for key, rates in factorizer.estimated():
    sorted_idx = np.argsort(rates)
    print(key, " : ", [items[x] for x in sorted_idx[:3]])
