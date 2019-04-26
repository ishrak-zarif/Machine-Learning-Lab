from itertools import permutations
import random
import copy
num_of_cities = 6

def initialize_population(num_of_cities):
    city = []
    for i in range(1,num_of_cities+1):
        city.append(i)
    cities = list(permutations(city))
    cities = cities[:100]
    return cities

def costMatrix(filename):
    cost_of_cities = []
    with open(filename) as file:
        for line in file:
            cost_of_city = []
            for val in line.split(" "):
                cost_of_city.append(int(val))
            cost_of_cities.append(cost_of_city)
    return cost_of_cities

def Fitness(solution,costMat):
    fit_values = []
    for sol in solution:
        fit = 0
        for i in range(0,len(sol)-1):
            fit += costMat[sol[i]-1][sol[i+1]-1]
            # print(sol[i]-1, ' ', sol[i+1]-1, ' ', fit)
        fit += costMat[sol[0]-1][sol[len(sol)-1]-1]
        fit_values.append(fit)
    return fit_values

def Selection(cities,fit_val):
    new_cities = []
    for i in range(len(fit_val)):
        j = random.randint(0,len(fit_val)-1)
        k = random.randint(0,len(fit_val)-1)
        if fit_val[j] < fit_val[k]:
            new_cities.append(cities[j])
        else:
            new_cities.append(cities[k])
    return new_cities

def Calculation(first, one, num_of_cities, Edge_Mat, val):
    if first == 0:
        if one[1] in Edge_Mat[val]:
            ind = Edge_Mat[val].index(one[1])
            Edge_Mat[val][ind] *= -1
        else:
            Edge_Mat[val].append(one[1])

        if one[num_of_cities - 1] in Edge_Mat[val]:
            ind = Edge_Mat[val].index(one[num_of_cities - 1])
            Edge_Mat[val][ind] *= -1
        else:
            Edge_Mat[val].append(one[num_of_cities - 1])

    elif first == num_of_cities - 1:
        if one[0] in Edge_Mat[val]:
            ind = Edge_Mat[val].index(one[0])
            Edge_Mat[val][ind] *= -1
        else:
            Edge_Mat[val].append(one[0])

        if one[num_of_cities - 2] in Edge_Mat[val]:
            ind = Edge_Mat[val].index(one[num_of_cities - 2])
            Edge_Mat[val][ind] *= -1
        else:
            Edge_Mat[val].append(one[num_of_cities - 2])
    else:
        if one[first - 1] in Edge_Mat[val]:
            ind = Edge_Mat[val].index(one[first - 1])
            Edge_Mat[val][ind] *= -1
        else:
            Edge_Mat[val].append(one[first - 1])

        if one[first + 1] in Edge_Mat[val]:
            ind = Edge_Mat[val].index(one[first + 1])
            Edge_Mat[val][ind] *= -1
        else:
            Edge_Mat[val].append((one[first + 1]))

def cross(one, two, num_of_cities):
    Edge_Mat = []
    for i in range(num_of_cities+1):
        Edge_Mat.append(list())
    for i in range(num_of_cities):
        first = one.index(i+1)
        second = two.index(i+1)
        Calculation(first, one, num_of_cities, Edge_Mat, i+1)
        Calculation(second, two, num_of_cities, Edge_Mat, i+1)

    check = [0]*(num_of_cities+1)
    resulted_sol = []
    i = 1
    j = 1
    term = 1
    while(i != num_of_cities+1):
        flag = 1
        for k in sorted(Edge_Mat[j]):
            if not check[abs(k)]:
                j = abs(k)
                resulted_sol.append(j)
                check[j] = 1
                i += 1
                flag = 0
                break
        if flag:
            j = 1
            while(j != num_of_cities):
                if not check[j]:
                    resulted_sol.append(j)
                    check[j] = 1
                    i += 1
                    break
                j += 1
        term += 1
        if term >= num_of_cities +5:
            break

    if len(resulted_sol)!= num_of_cities:
        for i in range(num_of_cities):
            if not check[i+1]:
                resulted_sol.append(i+1)
    return resulted_sol

def Crossover(selected_cities, num_of_cities):
    crossed_cities = []
    for i in range(len(selected_cities)):
        if i%2==0:
            crossed_cities.append(cross(selected_cities[i],selected_cities[i+1], num_of_cities))
    return crossed_cities

def Mutation(Crossed_cities):
    if random.randint(1,1000) == 2:
        i = random.randint(0,len(Crossed_cities)-1)
        j = random.randint(0,len(Crossed_cities[i])-1)
        k = random.randint(0,len(Crossed_cities[i])-1)
        Crossed_cities[i][j], Crossed_cities[i][k] = Crossed_cities[i][k], Crossed_cities[i][j]

def New_population(Prev_population, New_pop):
    for j in range(len(New_pop)):
        i = random.randint(0, len(Prev_population)-1)
        Prev_population[i] = New_pop[j]



Cities = initialize_population(num_of_cities)
costMat = costMatrix('CostMatrix.txt')
Fit_val = Fitness(Cities, costMat)

# print(Cities)
# print(costMat)

# print('Initial: ')
# for i in range(0, len(Cities)):
#     print(Cities[i], ' ', Fit_val[i])
Best_val = min(Fit_val)

for i in range(100):
    PseudoCities = copy.copy(Cities)
    Fit_val = Fitness(PseudoCities, costMat)
    Selected_cities = Selection(PseudoCities,Fit_val)
    Crossed_cities = Crossover(Selected_cities,num_of_cities)
    Mutation(Crossed_cities)
    New_fit_val = Fitness(Crossed_cities, costMat)

    # print('Iteration number : ', i)
    for j in range(0, len(Crossed_cities)):
        # print(Crossed_cities[j], ' ', New_fit_val[j])
        if (Best_val == New_fit_val[j]):
            print(Crossed_cities[j], ' ', New_fit_val[j])

    if min(New_fit_val) <= Best_val:
        Best_val = min(New_fit_val)
    New_population(Cities,Crossed_cities)
print('Best value : ', Best_val)