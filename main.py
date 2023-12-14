from random import random
from math import floor

mates:list = ['Thomas', 'Cosmin', 'Shan', 'Lilian']

equip1:list = []
equip2:list = []

equip1.append(mates.pop(floor(random() * len(mates))))
equip1.append(mates.pop(floor(random() * len(mates))))

equip2 = [x for x in mates]

print("Teams made :")
print(f' - Team 1 : {equip1}')
print(f' - Team 2 : {equip2}')
