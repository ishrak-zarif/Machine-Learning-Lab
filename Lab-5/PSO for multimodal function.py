import random

c1 = 1.5
c2 = 1.5
size = 20
iterations = 500
velocity_x = []
velocity_y = []
pos_x = []
pos_y = []
pBest_x = []
pBest_y = []
gBest_x = 0
gBest_y = 0

def fitness(x, y):
    return -(x * x) - (y*y) + 5

for i in range(size):
    pos_x.append(random.randint(0, 1))
    pos_y.append(random.randint(0, 1))
    velocity_x.append(random.randint(0, 1))
    velocity_y.append(random.randint(0, 1))
    pBest_x.append(pos_x[i])
    pBest_y.append(pos_y[i])

for i in range(iterations):
    for j in range(size):
        if fitness(pos_x[j], pos_y[j]) > fitness(pBest_x[j], pBest_y[j]):
            pBest_x[j] = pos_x[j]
            pBest_y[j] = pos_y[j]
    for k in range(size):
        if fitness(pBest_x[k], pBest_y[k]) > fitness(gBest_x, gBest_y):
            gBest_x = pBest_x[j]
            gBest_y = pBest_y[j]
    for t in range(size):
        velocity_x[t] = (velocity_x[t]) + (c1 * random.random() * (pBest_x[t] - pos_x[t])) + (c2 * random.random() * (gBest_x - pos_x[t]))
        pos_x[t] = pos_x[t] + velocity_x[t]
        velocity_y[t] = (velocity_y[t]) + (c1 * random.random() * (pBest_y[t] - pos_y[t])) + (c2 * random.random() * (gBest_y - pos_y[t]))
        pos_y[t] = pos_y[t] + velocity_y[t]

print("result : ", fitness(gBest_x, gBest_y))