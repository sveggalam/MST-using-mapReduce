from pyspark import SparkContext, SparkConf
from collections import defaultdict
import time

# Find operation in union-find with path compression
def find(x, parent):
    if parent[x] != x:
        parent[x] = find(parent[x], parent)  # Path compression
    return parent[x]

# Union operation in union-find with rank
def union(x, y, parent, rank):
    root_x = find(x, parent)
    root_y = find(y, parent)

    if root_x != root_y:
        if rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        elif rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

# Function to find the minimum edges for each component in a partition
def find_min_edges(partition, parent_bc):
    """
    For each partition of edges, find the minimum edge for each component.
    partition: List of edges.
    parent_bc: Broadcasted parent array for union-find.
    """
    parent = parent_bc.value
    min_edges = []  # To store the minimum edge for each component

    for src, dst, weight in partition:
        # Find the root component for both vertices
        root_src = find(src, parent)
        root_dst = find(dst, parent)

        if root_src != root_dst:  # Ignore edges within the same component
            # Always use the smaller of the two roots as the key
            component = min(root_src, root_dst)
            # Yield the minimum edge for this component
            min_edges.append((min(src,dst), (src, dst, weight)))
            min_edges.append((max(src,dst), (src,dst, weight)))
    return min_edges

# Borůvka's MST algorithm using PySpark
def boruvka_mst(edges_rdd, num_vertices, sc):
    """
    Borůvka's MST algorithm using PySpark.
    """
    # Initialize parent and rank for union-find
    parent = list(range(num_vertices))
    rank = [0] * num_vertices

    mst_edges = []  # Final MST edges

    while len(mst_edges) < num_vertices - 1:
        # Broadcast the parent array to all workers
        parent_bc = sc.broadcast(parent)

        # Step 1: Find the minimum edges for each component
        min_edges = (
            edges_rdd.mapPartitions(lambda partition: find_min_edges(partition, parent_bc))  # Local minimum edges
            .reduceByKey(lambda e1, e2: e1 if e1[2] < e2[2] else e2)  # Global minimum edge for each component
            .map(lambda x: x[1])  # Extract the edges (not the component key)
            .collect()
        )

        # Step 2: Add the minimum edges to the MST and union their components
        for src, dst, weight in min_edges:
            root_src = find(src, parent)
            root_dst = find(dst, parent)
            if root_src != root_dst:
                mst_edges.append((src, dst, weight))
                union(src, dst, parent, rank)

        # Step 3: Update the edges to reflect new components
        parent_bc = sc.broadcast(parent)  # Update broadcast
        edges_rdd = edges_rdd.mapPartitions(lambda partition: [
            (find(src, parent_bc.value), find(dst, parent_bc.value), weight) for src, dst, weight in partition
        ])

    return mst_edges

if __name__ == "__main__":
    # Initialize SparkContext
    conf = SparkConf().setAppName("BoruvkaMST").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    # Input the number of vertices and edges
    print("Enter the number of vertices:")
    num_vertices = int(input().strip())
    print("Enter the edges in the format 'src dst weight', one per line. Enter 'done' to finish:")
    parent = list(range(num_vertices))
    rank = [0] * num_vertices
    edges = []
    
    # Reading input edges
    while True:
        line = input().strip()
        if line.lower() == "done":
            break
        src, dst, weight = map(int, line.split())
        edges.append((src, dst, weight))

    # Convert edge list to RDD
    start_time = time.time()
    edges_rdd = sc.parallelize(edges)

    # Run Borůvka's algorithm to find the MST
    mst_edges = boruvka_mst(edges_rdd, num_vertices, sc)
    end_time = time.time()
    # Output the result
    print("\nMinimum Spanning Tree edges:")
    total_weight = sum(weight for _, _, weight in mst_edges)
    for src, dst, weight in mst_edges:
        print(f"Edge {src}-{dst}: {weight}")
    print(f"Total MST weight: {total_weight}")
    print(f"\nProgram executed in {end_time - start_time:.2f} seconds.")
    # Stop SparkContext
    sc.stop()
