import random

c1 = 1.5
c2 = 1.5
dimension = 20
iterations = 500

velocity = []
pos = []
pBest = []
gBest = 0

def fitness(x):
    return -(x * x) + 5

for i in range(dimension):
    pos.append(random.randint(0, 1))
    velocity.append(random.randint(0, 1))
    pBest.append(pos[i])

for i in range(iterations):
    for j in range(dimension):
        if fitness(pos[j]) > fitness(pBest[j]):
            pBest[j] = pos[j]
    gBest = fitness(max(pBest))
    for t in range(dimension):
        velocity[t] = (velocity[t]) + (c1 * random.random() * (pBest[t] - pos[t])) + (c2 * random.random() * (gBest - pos[t]))
        pos[t] = pos[t] + velocity[t]

print("result: ", gBest)