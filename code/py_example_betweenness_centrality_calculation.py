import networkx as nx
import matplotlib.pyplot as plt

# Create the full graph
G = nx.Graph()

# Add all nodes
nodes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
G.add_nodes_from(nodes)

# Add all edges
edges = [
    ('0', '1'), ('1', '3'), ('1', '4'), ('1', '5'),
    ('3', '2'), ('2', '6'), ('2', '7'), ('2', '8'), ('2', '9'), ('2', '10'), ('2', '11'), ('2', '12'), ('11', '12')
]
G.add_edges_from(edges)

def calculate_and_print_centrality(H, title):
    """Calculate and print betweenness centrality for the given graph."""
    betweenness = nx.betweenness_centrality(H, normalized=False)
    betweenness_norm = nx.betweenness_centrality(H, normalized=True)
    num_nodes = len(H.nodes)
    normalization_factor = (num_nodes - 1) * (num_nodes - 2) / 2  # For undirected graphs
    
    print(f"\n{title}:")
    print(f"Normalization Factor: {normalization_factor:.4f}")
    print("Raw Betweenness Centrality:")
    for node, value in betweenness.items():
        print(f"Node {node}: {value:.4f}")
    print("Normalized Betweenness Centrality:")
    for node, value in betweenness_norm.items():
        print(f"Node {node}: {value:.4f}")

    # # Draw the graph
    # plt.figure(figsize=(8, 6))
    # pos = nx.spring_layout(H)
    # nx.draw(
    #     H, pos, with_labels=True, node_color='skyblue', edge_color='gray',
    #     node_size=800, font_size=10, font_weight='bold'
    # )
    # plt.title(title)
    # plt.show()

# Visualize the full graph
# plt.figure(figsize=(10, 8))
# pos = nx.spring_layout(G)
# nx.draw(
#     G, pos, with_labels=True, node_color='lightgreen', edge_color='gray',
#     node_size=800, font_size=10, font_weight='bold'
# )
# plt.title("Full Graph")
# plt.show()

# Subgraph for Node 1's neighbors
subgraph_nodes_1 = ['0', '1', '3', '4', '5']
H1 = G.subgraph(subgraph_nodes_1)
calculate_and_print_centrality(H1, "Betweenness Centrality for Node 1's Neighbors")

# Subgraph for Node 5's neighbors
subgraph_nodes_5 = ['3', '2', '6', '7', '8', '9', '10', '11', '12']
H5 = G.subgraph(subgraph_nodes_5)
calculate_and_print_centrality(H5, "Betweenness Centrality for Node 5's Neighbors")

# Betweenness centrality for the full graph
full_graph_betweenness = nx.betweenness_centrality(G, normalized=False)
full_graph_betweenness_norm = nx.betweenness_centrality(G, normalized=True)
num_nodes = len(G.nodes)
normalization_factor = (num_nodes - 1) * (num_nodes - 2) / 2

print("\nFull Graph Betweenness Centrality:")
print(f"Normalization Factor: {normalization_factor:.4f}")
print(f"Node 1 (Raw): {full_graph_betweenness['1']:.4f}")
print(f"Node 1 (Normalized): {full_graph_betweenness_norm['1']:.4f}")
print(f"Node 5 (Raw): {full_graph_betweenness['5']:.4f}")
print(f"Node 5 (Normalized): {full_graph_betweenness_norm['5']:.4f}")
