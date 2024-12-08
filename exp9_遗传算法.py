# 使用遗传算法解决背包问题（Knapsack Problem）
from typing import List
import random
import matplotlib.pyplot as plt

# 生成30个物品的数据集
def generate_items(num_items: int):
    items = []
    for _ in range(num_items):
        weight = random.randint(1, 20)
        value = random.randint(10, 100)
        items.append((weight, value))
    return items

# 初始化种群
def initialize_population(population_size: int, num_items: int) -> List[List[int]]:
    population = []
    for _ in range(population_size):
        # 随机生成一个个体（二进制串表示）
        individual = [random.randint(0, 1) for _ in range(num_items)]
        population.append(individual)
    return population

# 选择父代
def selection(population, fitness_values, num_parents):
    # 计算适应度总和
    total_fitness = sum(fitness_values)
    # 计算每个个体的选择概率
    selection_probabilities = [fitness / total_fitness for fitness in fitness_values]
    # 选择num_parents个个体
    selected_parents = random.choices(population, weights=selection_probabilities, k=num_parents)
    return selected_parents

# 交叉
def crossover(parent1, parent2):
    # 随机选择交叉点
    crossover_point = random.randint(1, len(parent1) - 1)
    # 生成子代
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# 变异
def mutation(individual, mutation_rate):
    mutated_individual = individual[:]
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            # 将位翻转
            mutated_individual[i] = 1 - mutated_individual[i] # 1->0, 0->1
    return mutated_individual

# 适应度函数
def fitness_function(individual, items, max_weight):
    total_weight = 0
    total_value = 0
    for i in range(len(individual)):
        if individual[i] == 1:
            total_weight += items[i][0]
            total_value += items[i][1]
    # 如果总重量超过背包容量，则适应度为0
    if total_weight > max_weight:
        return 0
    else:
        return total_value
    
def genetic_algorithm(items, max_weight, population_size, num_generations, crossover_rate, mutation_rate):
    # 初始化种群
    population = initialize_population(population_size, len(items))
    best_fitness = 0
    best_individual = None
    current_weight = 0
    fitness_over_time = []

    for generation in range(num_generations):
        # 计算每个个体的适应度
        fitness_values = [fitness_function(individual, items, max_weight) for individual in population]

        # 选择父代
        num_parents = int(population_size * 0.5)  # 选择一半的个体作为父代
        selected_parents = selection(population, fitness_values, num_parents)

        # 生成子代
        offspring = []
        for i in range(0, num_parents, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            # 变异
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            offspring.extend([child1, child2])

        # 更新种群
        population = offspring
        fitness_values = [fitness_function(individual, items, max_weight) for individual in population]

        # 记录最优个体的适应度和总重量
        best_individual_index = fitness_values.index(max(fitness_values))
        current_weight = sum(items[i][0] for i in range(len(population[best_individual_index]))if population[best_individual_index][i] == 1)
        current_best_fitness = fitness_values[best_individual_index]
        current_best_individual = population[best_individual_index]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual

        # 记录适应度随迭代次数的变化
        fitness_over_time.append(best_fitness)

    return best_individual, best_fitness, fitness_over_time, current_weight

if __name__ == '__main__':
    # 生成物品数据集
    items = generate_items(30)
    # 输出数据集
    for i, (weight, value) in enumerate(items):
        print(f"Item {i+1}: weight={weight}, value={value}")
    # 设置算法参数
    max_weight = 200
    population_size = 100
    num_generations = 100
    crossover_rate = 0.8
    mutation_rate = 0.1
    # 运行遗传算法
    best_individual, best_fitness, fitness_over_time, current_weight = genetic_algorithm(items, max_weight, population_size, num_generations, crossover_rate, mutation_rate)
    # 输出结果
    print("Best individual:", best_individual)
    print("Best value:", best_fitness)
    print("Total weight:", current_weight)
    # 可视化适应度随迭代次数的变化
    plt.plot(fitness_over_time)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness/Value Over Generations')
    plt.show()