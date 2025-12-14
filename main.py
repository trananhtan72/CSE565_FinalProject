import sys
import os
from collections import defaultdict, deque

# --- 1. Graph Representation and Data Structures ---

class Graph:
    """Represents the directed graph and the flow."""
    def __init__(self, V, E, s, t):
        self.V = V  # Number of vertices
        self.E = E  # Number of edges
        self.s = s  # Source vertex (1-indexed)
        self.t = t  # Sink vertex (1-indexed)
        
        # Adjacency list: {u: [(v, [current_flow]), ...]}
        # Using a list [current_flow] to allow flow update by reference
        self.adj = defaultdict(list) 
        
        # Store initial flow for reference if needed
        self.initial_flow = {} 

    def add_edge(self, u, v, flow):
        """
        Adds a directed edge (u, v) with flow f(e).
        """
        flow_list = [flow] 
        self.adj[u].append((v, flow_list))
        self.initial_flow[(u, v)] = flow
        

# --- 2. Input/Output and File Handling ---

def parse_input(filepath):
    """
    Reads the .graph input file and builds the Graph structure.
    
    [cite_start]s is 1, t is |V|[cite: 28].
    """
    try:
        with open(filepath, 'r') as f:
            # Read |V| and |E|
            line = f.readline().split()
            if len(line) != 2:
                raise ValueError("First line must contain two integers: |V| and |E|.")
            V = int(line[0])
            E = int(line[1])
            
            s = 1 # Source is numbered 1 [cite: 28]
            t = V # Sink is numbered |V| [cite: 28]
            
            graph = Graph(V, E, s, t)
            
            # Read |E| edges
            for i in range(E):
                line = f.readline().split()
                if len(line) != 3:
                    continue 
                    
                u, v, flow = map(int, line)
                graph.add_edge(u, v, flow)

        return graph
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading or parsing input file: {e}", file=sys.stderr)
        sys.exit(1)


def write_output(filepath, paths, cycles):
    """
    Writes the decomposition solution to the .out file.
    
    Output Format:
    [cite_start]Line 1: |P| |C| [cite: 30]
    [cite_start]Next |P| lines: w(p) followed by vertices in p (space-separated) [cite: 31]
    [cite_start]Next |C| lines: w(c) followed by vertices in c (space-separated) [cite: 32]
    """
    try:
        with open(filepath, 'w') as f:
            # Line 1: |P| |C|
            f.write(f"{len(paths)} {len(cycles)}\n")
            
            # Write Paths
            for w, p_nodes in paths:
                path_str = " ".join(map(str, p_nodes))
                f.write(f"{w} {path_str}\n")
                
            # Write Cycles
            for w, c_nodes in cycles:
                cycle_str = " ".join(map(str, c_nodes))
                # Note: c_nodes should be the list of vertices in the cycle, 
                # [cite_start]e.g., (a, b, c, e) for c1 in the example[cite: 27].
                f.write(f"{w} {cycle_str}\n")
                
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)


# --- 3. Core Algorithm (Heuristic Flow Decomposition) ---

def find_path(graph):
    """
    Finds an s-t path with positive residual flow using BFS.
    Returns: (path_nodes, min_flow) or (None, 0)
    """
    s = graph.s
    t = graph.t
    
    queue = deque([s])
    parent = {s: None}
    
    while queue:
        u = queue.popleft()
        if u == t:
            break
            
        for v, flow_list in graph.adj[u]:
            current_flow = flow_list[0]
            if current_flow > 0 and v not in parent:
                parent[v] = u
                queue.append(v)
    else:
        return None, 0

    # Reconstruct path and find min flow
    path_nodes = []
    current = t
    min_flow = float('inf')
    
    while current is not None:
        path_nodes.append(current)
        if current != s:
            u = parent[current]
            for v, flow_list in graph.adj[u]:
                if v == current:
                    min_flow = min(min_flow, flow_list[0])
                    break
        current = parent.get(current) # Use .get() for safety at the end of path
    
    path_nodes.reverse()
    return path_nodes, int(min_flow) # Flow must be integral [cite: 5]


def find_cycle(graph):
    """
    Finds a cycle with positive residual flow using a path search v -> ... -> u for an edge (u, v).
    Returns: (cycle_nodes, min_flow) or (None, 0)
    """
    def find_path_back(start_node, target_node, initial_flow_edge):
        # We need a path v -> ... -> u
        queue = deque([start_node])
        parent = {start_node: None}
        
        while queue:
            curr = queue.popleft()
            if curr == target_node:
                break
            
            for next_node, flow_list in graph.adj[curr]:
                current_flow = flow_list[0]
                # Check for flow > 0 and not already visited
                if current_flow > 0 and next_node not in parent: 
                    parent[next_node] = curr
                    queue.append(next_node)
        
        if target_node not in parent:
            return None, 0 # No path found
        
        # Reconstruct path v -> ... -> u and find min flow along it
        path_nodes = []
        current = target_node
        min_flow = float('inf')
        
        while current is not None:
            path_nodes.append(current)
            if current != start_node:
                prev = parent[current]
                for next_node, flow_list in graph.adj[prev]:
                    if next_node == current:
                        min_flow = min(min_flow, flow_list[0])
                        break
            current = parent.get(current)
        
        path_nodes.reverse() # path_nodes is now [v, ..., u]
        
        # Include the initial edge (u, v) flow
        min_flow = min(min_flow, initial_flow_edge)
        
        return path_nodes, min_flow

    # Iterate over all edges (u, v) with flow > 0
    for u in range(1, graph.V + 1):
        for v, flow_list in graph.adj[u]:
            flow = flow_list[0]
            if flow > 0:
                # Try to find a path from v back to u
                path_nodes, min_flow = find_path_back(v, u, flow)
                
                if path_nodes is not None and min_flow > 0:
                    # Cycle found: u -> v -> ... -> u
                    # path_nodes is [v, ..., u]. The full cycle is [u] + path_nodes
                    full_cycle = [u] + path_nodes
                    
                    # [cite_start]The output format for a cycle is the list of vertices, e.g., (a, b, c, e) [cite: 27]
                    # Since our list is [u, v, ..., u], we return the vertices up to the second 'u'
                    return full_cycle[:-1], int(min_flow) # Exclude the repeated last node 'u'
                    
    return None, 0


def decompose_flow(graph):
    """
    Performs the flow decomposition into paths P and cycles C using the heuristic.
    """
    P = [] # List of (weight, path_nodes)
    C = [] # List of (weight, cycle_nodes)
    
    # 1. Path Decomposition (s-t paths)
    while True:
        path_nodes, weight = find_path(graph)
        
        if path_nodes is None or weight == 0:
            break
        
        P.append((weight, path_nodes))
        
        # Subtract the weight from the flow of every edge in the path
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            for next_v, flow_list in graph.adj[u]:
                if next_v == v:
                    flow_list[0] -= weight # Update flow
                    break
        
    # 2. Cycle Decomposition
    while True:
        cycle_nodes, weight = find_cycle(graph)
        
        if cycle_nodes is None or weight == 0:
            break
        
        # --- MODIFICATION START ---
        # cycle_nodes is (v1, v2, ..., vk) where vk -> v1 closes the loop.
        # To explicitly show the loop back, append the first vertex (v1) to the end.
        
        # Save the original start node
        start_node = cycle_nodes[0]
        
        # Append the start node to the end of the list
        explicit_cycle_nodes = cycle_nodes + [start_node]
        
        C.append((weight, explicit_cycle_nodes)) 
        
        # --- MODIFICATION END ---
        
        # Subtract the weight from the flow of every edge in the cycle
        # We must use the original cycle_nodes list for flow subtraction logic
        for i in range(len(cycle_nodes)):
            u = cycle_nodes[i]
            v = cycle_nodes[(i + 1) % len(cycle_nodes)] # next vertex in cycle (wraps around)
            
            for next_v, flow_list in graph.adj[u]:
                if next_v == v:
                    flow_list[0] -= weight # Update flow
                    break
                    
    return P, C


# --- 4. Main Execution Block ---

def main():
    """
    Handles command line arguments and file flow.
    Expected command: python3 main.py student_test_cases/NAME.graph
    """
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <input_file_path>", file=sys.stderr)
        sys.exit(1)
        
    input_filepath = sys.argv[1]
    
    # 1. Determine output file path and name
    
    # Get the base filename (e.g., NAME.graph)
    filename = os.path.basename(input_filepath)
    base_name = filename.rsplit('.', 1)[0] # Extract NAME
    
    # Output file must be generated in the 'outputs' directory
    output_filename = f"{base_name}.out"
    output_directory = "outputs"
    
    # Create the output directory if it doesn't exist 
    os.makedirs(output_directory, exist_ok=True)
    
    output_filepath = os.path.join(output_directory, output_filename)

    print(f"Processing input file: {input_filepath}")
    
    # 2. Parse input
    graph = parse_input(input_filepath)
    
    # --- Line to show how many files are read ---
    print(f"Successfully read 1 file.") 
    
    # 3. Decompose Flow
    paths, cycles = decompose_flow(graph)
    
    # 4. Write output
    write_output(output_filepath, paths, cycles)
    
    print(f"Generated output file: {output_filepath}")
    print("Decomposition complete.")

if __name__ == "__main__":
    # [cite_start]Ensure single-thread execution as required [cite: 35]
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    main()