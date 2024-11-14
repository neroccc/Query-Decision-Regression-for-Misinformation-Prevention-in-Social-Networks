import math
import os
import random
import sys
import math
import scipy
import scipy.integrate as integrate
import scipy.optimize as optimize
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import laplacian_kernel
import heapq
import multiprocessing
from functools import partial

def save_pretrain(path, weights, featureIndex, featurePath):
    with open(path + "/featureIndex", 'w') as outfile:
        for index, w in zip(featureIndex, list(weights)):
            outfile.write(str(index) + " " + str(w) + " " + "\n")
    outfile.close()


def count_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return len(lines)

def E_calculate(preTrainPath,featurePath,stoGraphPath):
    eNum = count_lines(stoGraphPath)
    w = [0] * eNum
    infile = open(preTrainPath + "featureIndex", 'r')
    outfolder = featurePath +"/"
    while True:
        line = infile.readline()
        if not line:
            infile.close()
            break
        ints = line.split()
        number = ints[0]
        weight = float(ints[1])
        inpath = "{}{}_graph.txt".format(outfolder, number)
        with open(inpath, 'r') as feature:
            i = 0
            while True:
                line = feature.readline()
                if not line:
                    feature.close()
                    break
                ints = line.split()
                node_weight = float(ints[2])
                if node_weight == float('inf'):
                    node_weight = 0
                w[i] = w[i] + weight * node_weight
                i = i + 1
    infile.close()
    infile = open(stoGraphPath, 'r')
    outfile = open(preTrainPath + "/Egraph", 'w')
    w = [float('inf') if x == 0 else x for x in w]
    mini_vaule = min(w)    
    nor_r = 3 / mini_vaule
    for i in range(len(w)):
        w[i] = w[i] * nor_r
    i = 0
    while True:
        line = infile.readline()
        if not line:
            infile.close()
            break
        ints = line.split()
        if w[i] != float('inf'):
         mean = math.ceil(float(w[i]))
         node1 = ints[0]
         node2 = ints[1]
         alpha = random.randint(1, 10)
         tag = 1
         while(tag != 0):
             try:
                 beta = estimate_beta(alpha, mean)
                 tag = 0
             except ValueError:
                 print(mean)
                 alpha = alpha + 1
         outfile.write(str(node1) + " ")
         outfile.write(str(node2) + " ")
         outfile.write(str(alpha) + " ")
         outfile.write(str(beta) + " ")
         outfile.write(str(mean) + "\n")
        i = i + 1
    outfile.close()
def estimate_beta(alpha, expected_time):
    # Define the distribution of time
    def time_dist(x, alpha, beta):
        return alpha * math.pow(-math.log(1 - x), beta)

    # Define the objective function to minimize
    def objective(beta):
        integrand = lambda x: time_dist(x, alpha, beta) * x ** beta * (-math.log(1 - x))
        estimated_time = integrate.quad(integrand, 0, 1)[0]
        return (estimated_time - expected_time) ** 2

    # Find the value of beta that minimizes the objective function
    result = optimize.minimize_scalar(objective, method='bounded', bounds=(0.01, 10))
    beta_estimate = result.x

    return math.ceil(beta_estimate)
def soup_generate(soup_number,path,graph,featureGenMethod,maxFeatureNum,featureNum,trainNum):
    folder_path = path + "/StratLearner/pre_train" + str(featureNum) +"_" +str(trainNum)
    items = os.listdir(folder_path)
    folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
    folders_value = len(folders)
    E_list = []
    folder_Nums = (np.random.permutation(folders_value))[0:soup_number]
    for item in folder_Nums:
        select_folder_path = os.path.join(folder_path, folders[item])
        Egraph = select_folder_path + "/Egraph"
        E_list.append(Egraph)
    data_path = path + "/data"
    featurePath = "{}/{}/feature/{}_{}/".format(data_path, graph, featureGenMethod, maxFeatureNum)
    genMultiRealization(maxFeatureNum/soup_number, featurePath, E_list, startIndex=0)
    return E_list

def genMultiRealization(num, outfolder, E_list, startIndex=0):
    r = len(E_list)
    num = int(num)
    for index in range(r):
        for cout in range(num):
            path = "{}{}_graph.txt".format(outfolder, num*index + cout + startIndex)
            genOneRealizationTrue(path,E_list[index])

def genOneRealizationTrue(outpath,index):
    with open(outpath, 'w') as outfile:
        infile = open(index, 'r')
        while True:
            line = infile.readline()
            if not line:
                infile.close()
                break
            ints = line.split()
            node1 = ints[0]
            node2 = ints[1]
            alpha = ints[2]
            beta = ints[3]
            mean = ints[4]
            weight = getWeibull(float(alpha), float(beta))
            outfile.write(node1 + " ")
            outfile.write(node2 + " ")
            outfile.write(str(weight) + "\n")
        outfile.close()

def getWeibull(alpha, beta):
    time = alpha * math.pow(-math.log(1 - random.uniform(0, 1)), beta)
    if time >= 0:
        return math.ceil(time) + 1
    else:
        sys.exit("time <0")
        return None

def dijkstra(graph, source):
    n = len(graph)
    distances = {i: float('inf') for i in range(n)}
    distances[source] = 0
    pq = [(0, source)]

    while pq:
        current_distance, u = heapq.heappop(pq)

        if current_distance > distances[u]:
            continue

        for v, weight in graph[u]:
            distance = current_distance + weight
            if distance < distances[v]:
                distances[v] = distance
                heapq.heappush(pq, (distance, v))

    return distances


def process_graph_file(graph_file, output_folder):
    # Parsing the graph data
    edges = []
    with open(graph_file, 'r') as f:
        for line in f:
            source, dest, weight = map(float, line.strip().split())
            edges.append((int(source), int(dest), weight))

    # Find the maximum node number to define the size of the graph
    max_node = max(max(source, dest) for source, dest, weight in edges)

    # Build the graph as an adjacency list
    graph = [[] for _ in range(max_node + 1)]
    for source, dest, weight in edges:
        graph[source].append((dest, weight))

    # Calculate shortest paths from each node using Dijkstra's algorithm
    all_pairs_distances = {}
    for node in range(max_node + 1):
        distances = dijkstra(graph, node)
        all_pairs_distances[node] = distances

    # Prepare the result for output
    result_lines = []
    for source in all_pairs_distances:
        for dest in all_pairs_distances[source]:
            distance = all_pairs_distances[source][dest]
            if not np.isinf(distance):  # Only write if distance is not infinite
                result_lines.append(f"{source} {dest} {distance}")

    # Derive output file path
    base_filename = os.path.basename(graph_file)
    number = base_filename.split("_")[0]
    distance_file_output = os.path.join(output_folder, f"{number}_distance.txt")

    # Write the results to the output file
    with open(distance_file_output, 'w') as f:
        f.write('\n'.join(result_lines))

    print(f"Processed {graph_file} and saved the distances to {distance_file_output}")


def dis_cal(path,maxFeatureNum):
    # Define the folder paths
    input_folder = path
    output_folder = path

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all graph files
    graph_files = [os.path.join(input_folder, f"{i}_graph.txt") for i in range(maxFeatureNum)]

    # Create a pool of workers
    with multiprocessing.Pool() as pool:
        pool.map(partial(process_graph_file, output_folder=output_folder), graph_files)
