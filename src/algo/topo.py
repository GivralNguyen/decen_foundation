import networkx as nx
import matplotlib.pyplot as plt
import random
import math
# Create a random graph with n users and p probability of connection
def create_random_graph(n, p, graph_name=None):
    # Generate the graph
    G = nx.erdos_renyi_graph(n, p)

    if graph_name != None:
        plt.figure(figsize=(5, 5))
        nx.draw(G, with_labels=True, node_color='lightcoral', node_size=600, font_size=10, pos=nx.circular_layout(G))
        plt.title(f"{graph_name} Topology with {n} Nodes and p={p}", fontsize=12)
        plt.savefig(graph_name, dpi=300, bbox_inches='tight')
    return G

# Create a randomized ring graph with n users arranged in a ring
def create_random_ring_graph(n, graph_name=None):
    # Create a list of nodes
    nodes = list(range(n))
    
    # Shuffle the nodes to randomize connections
    random.shuffle(nodes)
    
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes to the graph
    G.add_nodes_from(range(n))

    # Connect nodes in a ring topology
    for i in range(n):
        G.add_edge(nodes[i], nodes[(i+1) % n])
        
    if graph_name != None:
        plt.figure(figsize=(5, 5))
        nx.draw(G, with_labels=True, node_color='lightcoral', node_size=600, font_size=10, pos=nx.circular_layout(G))
        plt.title(f"{graph_name} Topology with {n} Nodes", fontsize=12)

        # ✅ Save the figure
        plt.savefig(graph_name, dpi=300, bbox_inches='tight')
    return G

# def create_ring_graph(n):
#     G = nx.Graph()
#     G.add_nodes_from(range(n))
#     for i in range(n):
#         G.add_edge(i, (i + 1) % n)  # wrap-around to form a ring
    return G
# Create a regular graph with n users and d degree of each node
def create_regular_graph(n, d, graph_name=None):
    # Generate the graph
    G = nx.random_regular_graph(d, n)
    
    if graph_name != None:
        # Draw the graph
        plt.figure(figsize=(5, 5))
        nx.draw(G, with_labels=True, node_color='lightcoral', node_size=600, font_size=10, pos=nx.circular_layout(G))
        plt.title(f"{graph_name} Topology with {n} Nodes and d={d}", fontsize=12)

        # ✅ Save the figure
        plt.savefig(graph_name, dpi=300, bbox_inches='tight')
    return G

def create_multi_star_graph(n, centers,graph_name=None):
    """
    Create a multi-center star topology.
    centers: list of hub node IDs, e.g. [0, 5]
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Connect each non-center node to its nearest center (round robin)
    for i in range(n):
        if i not in centers:
            # assign each non-center node to a random or round-robin center
            assigned_center = centers[i % len(centers)]
            G.add_edge(i, assigned_center)

    # Optionally connect the centers together (common in multi-hub systems)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            G.add_edge(centers[i], centers[j])
    if graph_name != None:
        plt.figure(figsize=(5, 5))
        nx.draw(G, with_labels=True, node_color='lightcoral', node_size=600, font_size=10, pos=nx.circular_layout(G))
        plt.title(f"{graph_name} Topology with {n} Nodes", fontsize=12)

        # ✅ Save the figure
        plt.savefig(graph_name, dpi=300, bbox_inches='tight')

    return G

def create_exponential_graph(n, directed=True, graph_name=None):
    """
    Create a directed exponential graph as described in SGP experiments.
    Each node i connects to nodes (i + 2^k) % n for k = 0..⌊log2(n-1)⌋.
    """
    # Create directed or undirected graph
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))

    max_k = int(math.floor(math.log2(n - 1)))  # maximum hop exponent
    for i in range(n):
        for k in range(max_k + 1):
            j = (i + 2**k) % n
            if i != j:
                G.add_edge(i, j)
    if graph_name != None:
        plt.figure(figsize=(5, 5))
        nx.draw(G, with_labels=True, node_color='lightcoral', node_size=600, font_size=10, pos=nx.circular_layout(G))
        plt.title(f"{graph_name} Topology with {n} Nodes", fontsize=12)

        # ✅ Save the figure
        plt.savefig(graph_name, dpi=300, bbox_inches='tight')
    return G

def create_double_ring_graph(m=5, bridge=(0, 5), graph_name=None):
    """
    Create two ring (cycle) graphs of size m each,
    connected by one bridge edge.
    
    Parameters:
        m : int
            Number of nodes in each ring
        bridge : tuple(int, int)
            Nodes to connect between rings (default: node 0 of first ring to node 0 of second ring)
        graph_name : str or None
            Optional file name to save figure
    """
    # Create two separate cycles
    G1 = nx.cycle_graph(m)
    G2 = nx.cycle_graph(m)
    
    # Relabel second ring nodes to continue numbering (e.g., 0..m-1 and m..2m-1)
    G2 = nx.relabel_nodes(G2, lambda x: x + m)
    
    # Combine them
    G = nx.compose(G1, G2)
    
    # Add bridge edge between the two rings
    G.add_edge(bridge[0], bridge[1] + m)
    
    # Draw
    plt.figure(figsize=(5, 2))
    pos = {}
    # layout left ring
    pos.update(nx.circular_layout(range(m), scale=1.0, center=(-1.5, 0)))
    # layout right ring
    pos.update(nx.circular_layout(range(m, 2*m), scale=1.0, center=(1.5, 0)))
    
    nx.draw(G, pos, with_labels=True, node_color="teal", edge_color="skyblue",
            node_size=500, font_size=8, width=2)
    plt.title(f"Double Ring Graph (m={m})")
    
    if graph_name:
        plt.savefig(graph_name, dpi=300, bbox_inches="tight")
        print(f"✅ Saved to {graph_name}")
    else:
        plt.show()
    
    return G
def create_cluster_topology(n, n_clusters=3, intra_p=0.8, inter_p=0.05, graph_name=None):
    """
    Create a clustered (community-based) topology similar to NTK-DFL's 'clustered graph'.

    Args:
        n (int): total number of nodes
        n_clusters (int): number of clusters (communities)
        intra_p (float): probability of edge *within* a cluster (dense)
        inter_p (float): probability of edge *between* clusters (sparse)
        graph_name (str): optional name for saving the graph image
    Returns:
        nx.Graph: the generated clustered network
    """
    # Create an empty graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # --- Assign nodes to clusters
    nodes = list(range(n))
    random.shuffle(nodes)
    cluster_size = math.ceil(n / n_clusters)
    clusters = [nodes[i*cluster_size:(i+1)*cluster_size] for i in range(n_clusters)]

    # --- Intra-cluster edges (dense connections)
    for cluster in clusters:
        for i in cluster:
            for j in cluster:
                if i < j and random.random() < intra_p:
                    G.add_edge(i, j)

    # --- Inter-cluster edges (sparse links between clusters)
    for c1 in range(n_clusters):
        for c2 in range(c1 + 1, n_clusters):
            for i in clusters[c1]:
                for j in clusters[c2]:
                    if random.random() < inter_p:
                        G.add_edge(i, j)

    # --- Visualization (optional)
    if graph_name is not None:
        plt.figure(figsize=(5, 5))
        pos = nx.spring_layout(G, seed=42)
        colors = []
        for i in range(n):
            # color each cluster differently
            for c, cluster in enumerate(clusters):
                if i in cluster:
                    colors.append(plt.cm.tab10(c % 10))
        nx.draw(G, pos, with_labels=True, node_color=colors,
                node_size=600, font_size=10, edge_color="gray")
        plt.title(f"{graph_name}: {n_clusters} Clusters", fontsize=12)
        plt.savefig(graph_name, dpi=300, bbox_inches="tight")
        plt.close()

    return G
def create_random_line_graph(n, graph_name=None):
    """
    Create a randomized line topology:
    - Nodes are randomly shuffled.
    - Connected sequentially in a chain (no wrap-around).

    Args:
        n (int): number of nodes
        graph_name (str, optional): filename to save the visualization

    Returns:
        nx.Graph: the generated random line graph
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import random

    # Shuffle node order
    nodes = list(range(n))
    random.shuffle(nodes)

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Connect sequentially (no wrap-around)
    for i in range(n - 1):
        G.add_edge(nodes[i], nodes[i + 1])

    # Optional visualization
    if graph_name is not None:
        plt.figure(figsize=(5, 2))
        pos = {nodes[i]: (i, 0) for i in range(n)}  # straight-line layout
        nx.draw(G, pos, with_labels=True,
                node_color='lightcoral', node_size=600,
                font_size=10, edge_color='gray')
        plt.title(f"Random Line Topology with {n} Nodes", fontsize=12)
        plt.axis('off')
        plt.savefig(graph_name, dpi=300, bbox_inches='tight')
        plt.close()

    return G

# --- DEMO ---
if __name__ == "__main__":
    n = 10
    G = create_regular_graph(n,5,graph_name="regular_graph.png")
    G = create_random_graph(n,0.4,"random_graph.png")
    G = create_random_ring_graph(n,"ring.png")
    G = create_multi_star_graph(n,[5],"star.png")
    G = create_exponential_graph(n,True,"exponential.png")
    G = create_double_ring_graph(m=5, bridge=(0, 0), graph_name="double_ring.png")
    G = create_cluster_topology(n, n_clusters=3, intra_p=0.9, inter_p=0.05, graph_name="clustered_graph.png")
    G = create_random_line_graph(n, graph_name="line_topology.png")
    # Print edges
    print("Edges in exponential graph:")
    for e in G.edges():
        print(e)
    neighbors = list(G.neighbors(1))
    local_group = neighbors + [1]
    plt.close()