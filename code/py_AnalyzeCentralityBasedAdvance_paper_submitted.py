import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.ticker as ticker
import os
import sys
from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing import Pool, cpu_count
from itertools import product
import csv
from collections import deque, defaultdict
import time
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats as st
from scipy.stats import norm
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'lmodern',
    # "font.serif": 'Times',
    'font.size': 30,
    'text.usetex': True,
    'pgf.rcfonts': False,
    # 'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.latex.preamble': r'\usepackage{lmodern}'
})

def confidence_interval_t(data, confidence=0.95):
    """
    Calculate the t-distribution based confidence interval for a given dataset and confidence level.
    """
    data_array = 1.0 * np.array(data)
    degree_of_freedom = len(data_array) - 1
    sample_mean, sample_standard_error = np.mean(data_array), st.sem(data_array)
    t = st.t.ppf((1 + confidence) / 2., degree_of_freedom)
    margin_of_error = sample_standard_error * t
    confidence_interval = np.array([sample_mean - margin_of_error, sample_mean + margin_of_error])
    return sample_mean, confidence_interval, margin_of_error

def confidence_interval_normal(data, confidence=0.95):
    """
    Calculate the normal distribution based confidence interval for a given dataset and confidence level.
    """
    data_array = 1.0 * np.array(data)
    sample_mean, sample_standard_error = np.mean(data_array), st.sem(data_array)
    z = norm().ppf((1 + confidence) / 2.)
    margin_of_error = sample_standard_error * z
    confidence_interval = np.array([sample_mean - margin_of_error, sample_mean + margin_of_error])
    return sample_mean, confidence_interval, margin_of_error

def confidence_interval_init(data, confidence=0.95):
    """
    Initialize confidence interval calculations for a dataset, handling multidimensional data and selecting
    the appropriate method based on sample size.
    """
    data_array = 1.0 * np.array(data)
    dimensions = data_array.shape
    if len(dimensions) > 1:
        rows, columns = dimensions[0], dimensions[1]
        if columns <= 30:
            method = confidence_interval_t
        else:
            method = confidence_interval_normal
        sample_mean_array, confidence_interval_array, margin_of_error_array = method(data_array[0], confidence)
        for row in range(1, rows):
            sample_mean_new_row, confidence_interval_new_row, margin_of_error_new_row = method(data_array[row], confidence)
            sample_mean_array = np.append(sample_mean_array, sample_mean_new_row)
            confidence_interval_array = np.vstack((confidence_interval_array, confidence_interval_new_row))
            margin_of_error_array = np.append(margin_of_error_array, margin_of_error_new_row)
        return sample_mean_array, confidence_interval_array, margin_of_error_array
    else:
        if len(data_array) <= 30:
            return confidence_interval_t(data_array, confidence)
        else:
            return confidence_interval_normal(data_array, confidence)

def calculate_pairwise_distances(points, selected_points):
    """
    Calculate pairwise Euclidean distances between two lists of points.
    """
    points_array = np.array([p['pos'] for p in points])
    selected_points_array = np.array([sp['pos'] for sp in selected_points])
    return euclidean_distances(points_array, selected_points_array)

def calculate_distance(pos1, pos2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def create_network(area_length, area_side, num_nodes, a2a_comm_range, a2g_comm_range, GS_id, GS_pos):
    # Create positions in a dictionary format
    # positions = {i: (random.uniform(0, area_length), random.uniform(0, area_side)) for i in range(num_nodes)}
    positions = {i: (np.random.uniform(0, area_length), np.random.uniform(0, area_side)) for i in range(num_nodes)}

    # Initialize the graph
    G = nx.Graph()
    G.add_nodes_from(positions.keys())
    for i, pos in positions.items():
        for j, pos2 in positions.items():
            if i < j and np.hypot(pos[0] - pos2[0], pos[1] - pos2[1]) <= a2a_comm_range:
                G.add_edge(i, j)

    # Add the GS_id node with its specified position
    G.add_node(GS_id, pos=GS_pos)
    positions[GS_id] = GS_pos  # Include GS position in the positions dictionary

    # Connect GS_id to other nodes based on a2g_comm_range
    for i, pos in positions.items():
        if i != GS_id and np.hypot(GS_pos[0] - pos[0], GS_pos[1] - pos[1]) <= a2g_comm_range:
            G.add_edge(GS_id, i)

    return G, positions

def select_EFF_neighbors(neighbor_attr, points, m, R=100, threshold = None):
    """
    Selects up to m neighbors based on the farthest-first traversal strategy from a given point's position attribute,
    with a maximum radius R and a dynamic threshold for minimal selection distance.
    
    Args:
        neighbor_attr (dict): Attributes of the central node, must include 'pos' for position.
        points (list): A list of points with position 'pos' and identifier 'id'.
        m (int): Maximum number of points to select.
        R (float): The reference distance radius for calculating the default threshold.
        threshold (float, optional): Distance threshold for selecting neighbors. If None, it is set to R / sqrt(2).
    
    Returns:
        list: List of selected point IDs based on the farthest-first criteria.
    """    
    # Assign default threshold if None
    if threshold is None:
        threshold = R / np.sqrt(2)  # Example default based on R
      
    threshold = 1 / np.sqrt(2) * R
    # threshold = 1 / 2 * R
    # threshold = (np.sqrt(3) / 2) * R
    if not points:
        return []
    if len(points) <= m:
        return [point['id'] for point in points if 'id' in point]
    
    neighbor_pos = np.array(neighbor_attr['pos'])
    points = np.array(points)  # Ensure points is a NumPy array for easier manipulation
    selected_indices = set()
    selected_points = []

    random_index = random.randint(0, len(points) - 1)
    selected_indices.add(random_index)
    selected_points.append(points[random_index])

    while len(selected_points) < m:
        remaining_indices = [i for i in range(len(points)) if i not in selected_indices]
        if not remaining_indices:
            break  # Break if there are no remaining points to consider

        remaining_points = points[remaining_indices]
        distances_to_selected = calculate_pairwise_distances(remaining_points, selected_points)
        min_distances = np.min(distances_to_selected, axis=1)

        farthest_point_idx = np.argmax(min_distances)
        selected_idx = remaining_indices[farthest_point_idx]
        farthest_point = points[selected_idx]

        # Assuming farthest_point and neighbor_pos are both numpy arrays 
        distance_to_center = np.linalg.norm(np.array(farthest_point['pos']) - neighbor_pos)

        distance_to_selected_points = np.min(calculate_pairwise_distances([farthest_point], selected_points), axis=1)[0]
        if distance_to_selected_points > threshold and distance_to_center > 1 / np.sqrt(2) * threshold:
            selected_indices.add(selected_idx)
            selected_points.append(farthest_point)
        else:
            break

    selected_points_ids = [point['id'] for point in selected_points if 'id' in point]
    return selected_points_ids

def apply_eff_algorithm(G, positions, m_values, comm_range):
    """
    Apply EFF logic to determine subsets and neighbors for 1-hop, 2-hop, and 3-hop neighborhoods.
    """
    result_subset = {m: {node: {"S1": set(), "S2": set()} for node in G.nodes()} for m in m_values}
    result_neighbors = {m: {node: {"1-hop": set(), "2-hop": set(), "3-hop": set()} for node in G.nodes()} for m in m_values}
    # Step 1: Find 1-hop neighbors and calculate S^1_i for each node
    for m in m_values:
        for node in G.nodes():
            neighbors = set(nx.neighbors(G, node))
            result_neighbors[m][node]["1-hop"] = {(neighbor, neighbor) for neighbor in neighbors}
            neighbor_attrs = [{"id": neighbor, "pos": positions[neighbor]} for neighbor in neighbors]
            subset_S1 = set(select_EFF_neighbors({"pos": positions[node]}, neighbor_attrs, m, comm_range))
            result_subset[m][node]["S1"] = subset_S1

    # Step 2: Combine S^1_j for direct neighbors of each node to calculate 2-hop neighbors
    for m in m_values:
        for node in G.nodes():
            for neighbor, _ in result_neighbors[m][node]["1-hop"]:
                # Directly from the direct neighbor's S^1
                S1_neighbors = result_subset[m][neighbor]["S1"]
                filtered_2_hop = {(n, neighbor) for n in S1_neighbors if n != node and n not in {x[0] for x in result_neighbors[m][node]["1-hop"]}}
                result_neighbors[m][node]["2-hop"].update(filtered_2_hop)

    # Step 3: Calculate S^2_i for each node using remaining m-values
    for m in m_values:
        for node in G.nodes():
            if len(result_subset[m][node]["S1"]) < m:
                remaining_m = m - len(result_subset[m][node]["S1"])
                two_hop_candidates = [n for n, _ in result_neighbors[m][node]["2-hop"]]
                two_hop_attrs = [{"id": candidate, "pos": positions[candidate]} for candidate in two_hop_candidates]
                subset_S2 = set(select_EFF_neighbors({"pos": positions[node]}, two_hop_attrs, remaining_m, comm_range, threshold = 0))
                result_subset[m][node]["S2"].update(subset_S2)

    # Step 4: Combine S^2_j for direct neighbors of each node to calculate 3-hop neighbors
    for m in m_values:
        for node in G.nodes():
            for neighbor, _ in result_neighbors[m][node]["1-hop"]:
                # Directly from the direct neighbor's S^2
                S2_neighbors = result_subset[m][neighbor]["S2"]
                filtered_3_hop = {(n, neighbor) for n in S2_neighbors if n != node and n not in {x[0] for x in result_neighbors[m][node]["1-hop"]} and n not in {x[0] for x in result_neighbors[m][node]["2-hop"]}}
                result_neighbors[m][node]["3-hop"].update(filtered_3_hop)
    
    # Final processing to filter based on farthest 'via'
    final_neighbors = {m: {node: {"1-hop": set(), "2-hop": set(), "3-hop": set()} for node in G.nodes()} for m in m_values}
    
    for m in m_values:
        for node in G.nodes():
            for hop in ["1-hop", "2-hop", "3-hop"]:
                neighbors_via = result_neighbors[m][node][hop]
                if not neighbors_via:
                    continue

                # Group by neighbor and collect vias and distances
                neighbor_via_distances = {}
                for neighbor, via in neighbors_via:
                    distance = calculate_distance(positions[via], positions[node])
                    if neighbor not in neighbor_via_distances:
                        neighbor_via_distances[neighbor] = [(via, distance)]
                    else:
                        if neighbor_via_distances[neighbor][0][1] < distance:
                            neighbor_via_distances[neighbor] = [(via, distance)]
                        elif neighbor_via_distances[neighbor][0][1] == distance:
                            neighbor_via_distances[neighbor].append((via, distance))

                # Create the final set with randomly selected farthest 'via' per neighbor
                final_neighbors[m][node][hop] = {
                    (neighbor, random.choice(neighbor_via_distances[neighbor])[0])
                    for neighbor in neighbor_via_distances
                }
    
    return final_neighbors

def process_all_nodes_with_betweenness_with_subset(G, positions, m_values, comm_range, GS_id=501):
    """
    Process all nodes with betweenness and subsets generated by EFF logic.

    """
    # Create a copy of the graph to work with
    G_copy = G.copy()

    # Remove the GS_id node if it exists in the copy
    if GS_id in G_copy.nodes:
        G_copy.remove_node(GS_id)

    # Call apply_eff_algorithm to compute final neighbors
    final_neighbors = apply_eff_algorithm(G_copy, positions, m_values, comm_range)

    # Precompute distances to GS (common for all m-values)
    gs_pos = positions[GS_id]
    distances_to_GS = {
        node: np.linalg.norm(np.array(positions[node]) - np.array(gs_pos))
        for node in G_copy.nodes()
    }

    # Precompute degree for all nodes
    node_degrees = {node: G_copy.degree(node) for node in G_copy.nodes()}

    # Initialize results dictionary
    all_results = {m: {} for m in m_values}

    # Process each m-value
    for m in m_values:
        for node in G_copy.nodes():
            if node == GS_id:
                continue  # Skip GS_id node

            # Extract neighbors for 1-hop, 2-hop, and 3-hop
            neighbors_m = set()
            for hop in ['1-hop', '2-hop', '3-hop']:
                neighbors_m.update({neighbor for neighbor, _ in final_neighbors[m][node][hop]})

            # Create the subgraph for the subset
            H_subset_m = G_copy.subgraph(neighbors_m | {node}).copy()

            # Compute local betweenness centrality in the subset
            betweenness_m = nx.betweenness_centrality(H_subset_m, normalized=False)
            local_betweenness_m = betweenness_m.get(node, 0)

            # Normalize betweenness (optional)
            n_nodes_subset = len(H_subset_m.nodes)
            # normalization_factor = (n_nodes_subset - 1) * (n_nodes_subset - 2)
            # if normalization_factor > 0:
            #     local_betweenness_m /= normalization_factor
            # local_betweenness_m += 1

            # Prepare neighbors info
            neighbors_info_m = {
                '1-hop': list(final_neighbors[m][node]['1-hop']),
                '2-hop': list(final_neighbors[m][node]['2-hop']),
                '3-hop': list(final_neighbors[m][node]['3-hop'])
            }

            # Store results
            all_results[m][node] = {
                'subgraph': H_subset_m,
                'neighbors_info': neighbors_info_m,
                'local_betweenness': local_betweenness_m,
                'distance_to_GS': distances_to_GS[node],  # Precomputed
                'degree': node_degrees[node]  # Precomputed
            }

    return all_results

def process_all_nodes_with_betweenness_and_distance(G, positions, max_radius=1, GS_id=501):
    """
    Applies enhanced ego_graph analysis to every node except the destination node in the graph G,
    returning subgraphs, neighbor details, local betweenness centrality for each node, and
    Euclidean distance to the specified destination node (GS_id).

    Parameters:
    ----------
    G : graph
        A NetworkX Graph or DiGraph.

    max_radius : int, optional
        The maximum number of hops (radius) to include in subgraph neighborhoods.

    GS_id : int
        The node id of the destination node for distance calculations.

    Returns:
    -------
    dict: A dictionary with each node as keys except GS_id. Each value contains:
        - subgraph: The subgraph induced by neighbors within max_radius.
        - neighbors_info: Dictionary containing neighbor information classified by hop distance.
        - local_betweenness: Betweenness centrality for the node in the max_radius neighborhood.
        - euclidean_distance_to_GS: The Euclidean distance from the node to GS_id.
    
    Note:
    This function calculates betweenness centrality up to the specified max_radius,
    not incrementally from 1 to max_radius. It provides a single betweenness centrality
    measure considering all paths within the max_radius, not separate measures for each hop.
    """
    all_results = {}

    for n in G.nodes():
        if n == GS_id:
            continue  # Skip processing for the destination node

        # Compute Euclidean distance to the destination node (GS_id)
        gs_pos = positions[GS_id]
        n_pos = positions[n]
        distance_to_GS = np.linalg.norm(np.array(n_pos) - np.array(gs_pos))

        # Calculate shortest paths and create subgraph for max_radius
        sp_paths = nx.single_source_shortest_path(G, n, cutoff=max_radius)
        subgraph_nodes = set(sp_paths.keys()) - {GS_id}  # Exclude GS_id from subgraph
        H_max = G.subgraph(subgraph_nodes).copy()  # max_radius subgraph

        # Calculate local betweenness centrality for the subgraph
        betweenness_max = nx.betweenness_centrality(H_max, normalized=False)
        local_betweenness = betweenness_max.get(n, 0)  # Get betweenness centrality of node n

        # Collect neighbor information for max_radius
        neighbors_info = {f'{i}-hop': [] for i in range(1, max_radius + 1)}
        for node, path in sp_paths.items():
            if node != n and node != GS_id:  # Exclude the center and GS_id from neighbor info
                num_hops = len(path) - 1
                if num_hops <= max_radius:
                    via_node = path[1] if len(path) > 1 else None
                    hop_key = f'{num_hops}-hop'
                    neighbors_info[hop_key].append((node, via_node))

        # Compute network size for normalization of betweenness centrality
        n_nodes_max = len(subgraph_nodes)
        normalization_factor = (n_nodes_max - 1) * (n_nodes_max - 2)
        if normalization_factor > 0:
            local_betweenness /= normalization_factor

        # Add 1 to avoid betweenness = 0 and adjust scaling
        if normalization_factor > 0:
            local_betweenness = ((local_betweenness * normalization_factor) + 1)

        # Store the results
        all_results[n] = {
            'subgraph': H_max,
            'neighbors_info': neighbors_info,
            'local_betweenness': local_betweenness,
            'distance_to_GS': distance_to_GS
        }

    return all_results

def best_advance_within_k_hops(current, all_results, a2a_comm_range, k):
    """
    Finds the best node to advance within `k` hops.
    
    Parameters:
    ----------
    current : int
        The current node.
    all_results : dict
        A dictionary containing details for each node.
    a2a_comm_range : float
        The communication range for A2A communication.
    k : int
        The number of hops to consider.

    Returns:
    -------
    int: The node to advance towards, or None if no advancement found.
    """
    current_to_dest_distance = all_results[current]['distance_to_GS']
    best_node = None
    best_advance = k * a2a_comm_range  # Initialize with maximum advancement possible
    
    # Combine all hops up to k-hops into a single loop
    neighbors = (neighbor_info for hop_type in [f'{i}-hop' for i in range(1, k+1)]
                                for neighbor_info in all_results[current]['neighbors_info'][hop_type])
    
    for neighbor, via_node in neighbors:
        neighbor_to_dest_distance = all_results[neighbor]['distance_to_GS']
        # Calculate the geographic advancement
        advance = current_to_dest_distance - neighbor_to_dest_distance + k * a2a_comm_range

        if advance > best_advance:
            best_advance = advance
            best_node = via_node  # Use the via node as the best node to advance toward

    return best_node

def subset_betweenness_centrality_routing(current, all_results, all_results_subset, m):
    """
    Selects the next node in routing based on local betweenness centrality from a subset
    of neighbors within 1-hop, 2-hop, and 3-hop, combining both centrality-based and geographic advance approach.

    Parameters:
    ----------
    current : int
        The current node's identifier.
    all_results : dict
        Complete results containing node details including neighbors for fallback.
    all_results_subset : dict
        Subset of results with limited neighbor information.
    m : int
        Identifier for the subset size to consider for routing.

    Returns:
    -------
    int or None:
        The identifier of the next node to route to, or None if no suitable node is found.
    """
    current_to_dest_distance = all_results_subset[m][current]['distance_to_GS']

    # # Primary neighbors: only 1-hop
    # neighbors = (neighbor_info for neighbor_info in all_results_subset[m][current]['neighbors_info']['1-hop'])

    # Combine all neighbors from 1-hop only
    neighbors = all_results[current]['neighbors_info']['1-hop']

    # Fallback neighbors: 1-hop, 2-hop, and 3-hop
    neighbors_advance = (neighbor_info for hop_type in ['1-hop', '2-hop', '3-hop']
                         for neighbor_info in all_results_subset[m][current]['neighbors_info'][hop_type])

    # Try q-hop betweenness-based routing (1-hop only)
    max_betweenness = -float('inf')
    next_node = None

    for neighbor, via_node in neighbors:
        neighbor_to_dest_distance = all_results_subset[m][neighbor]['distance_to_GS']
        advance = current_to_dest_distance - neighbor_to_dest_distance

        if advance > 0:
            neighbor_betweenness = all_results_subset[m][neighbor]['local_betweenness']
            if neighbor_betweenness > max_betweenness:
                max_betweenness = neighbor_betweenness
                next_node = via_node

    # If no suitable node is found, fallback to max advance using 1-hop, 2-hop, and 3-hop
    if next_node is None:
        max_advance = -float('inf')
        for neighbor, via_node in neighbors_advance:
            neighbor_to_dest_distance = all_results_subset[m][neighbor]['distance_to_GS']
            advance = current_to_dest_distance - neighbor_to_dest_distance

            if advance > max_advance:
                max_advance = advance
                next_node = via_node

    return next_node

def BCGR(current, all_results, a2a_comm_range, k):
    """
    Implements Betweenness Centrality-based Geographic Routing (BCGR), selecting a neighbor
    with maximum local betweenness among those that advance closer to the destination.

    Parameters:
    ----------
    current : int
        The current node's identifier.
    all_results : dict
        A dictionary of nodes containing neighbor and betweenness details.
    a2a_comm_range : float
        The communication range, affecting which nodes are considered neighbors.
    k : int
        Maximum number of hops considered for neighbor selection, typically set to 1 for BCGR.

    Returns:
    -------
    int or None:
        The identifier of the next node to route to, or None if no suitable node is found.
    """
    k = 1  # Fixed number of hops to consider
    current_to_dest_distance = all_results[current]['distance_to_GS']
    
    # Combine all neighbors from 1-hop only
    neighbors = all_results[current]['neighbors_info']['1-hop']
    
    max_betweenness = -float('inf')
    next_node = None

    for neighbor, via_node in neighbors:
        neighbor_to_dest_distance = all_results[neighbor]['distance_to_GS']
        
        # Calculate the geographic advancement
        advance = current_to_dest_distance - neighbor_to_dest_distance
        
        # Only consider nodes with higher advancement than the current node
        if advance > 0:
            neighbor_betweenness = all_results[neighbor]['local_betweenness']
            if neighbor_betweenness > max_betweenness:
                max_betweenness = neighbor_betweenness
                next_node = via_node  # Use the via node for routing
    
    return next_node

def perform_geographic_routing(G, all_results, a2a_comm_range, a2g_comm_range, destination, find_next_hop, k):
    """
    Performs geographic routing from each node in the graph to a designated destination node.
    This function uses caching mechanisms to improve efficiency and calculates the hop stretch factor.

    Parameters:
    ----------
    G : networkx.Graph
        The graph over which nodes are routed.
    all_results : dict
        Contains details for each node including distances and neighbor information.
    a2a_comm_range : float
        The air-to-air communication range.
    a2g_comm_range : float
        The air-to-ground communication range.
    destination : int
        The identifier of the destination node.
    find_next_hop : function
        A callable that determines the best next hop in the routing process.
    k : int
        Maximum number of hops to consider (unused here but could be integrated in future).

    Returns:
    -------
    int:
        The number of successful routes achieved in the simulation.
    """
    successful_routes = 0
    path_cache = {}  # Cache for successful paths
    failed_path_cache = {}  # Cache for paths known to fail
    num_nodes = G.number_of_nodes() - 1

    # Compute hop counts from each node to the destination using Dijkstra
    hop_counts_to_dest = nx.single_source_shortest_path_length(G, destination)
    
    # Initialize the next hop dictionary with -1 for all nodes except the destination

    for node in G.nodes:
        if node != destination:
            try:
                all_dijkstra_paths = list(nx.all_shortest_paths(G, node, destination))
                dijkstra_next_hops = list(set(path[1] for path in all_dijkstra_paths))
                dijkstra_path_length = len(all_dijkstra_paths[0]) - 1  # Hop count in Dijkstra path

            except nx.NetworkXNoPath:
                dijkstra_next_hops = None
                dijkstra_path_length = None
                geographic_next_hop = None
                failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
                continue  # Move to the next node since no path exists

            current = node
            path = [current]  # Initialize path with the current node

            # Compute the path if not in cache
            while current != destination:
                path_key = (current, destination)

                # Check failed path cache first
                if path_key in failed_path_cache:
                    failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
                    break  # If intermediate path is known to fail, stop processing

                # Attempt to use the successful path cache
                if path_key in path_cache:
                    cached_subpath = path_cache[path_key]
                    if path and cached_subpath and path[-1] == cached_subpath[0]:
                        path.extend(cached_subpath[1:])
                    else:
                        path.extend(cached_subpath)
                    if len(path) != len(set(path)):
                        print("(A) When you find a cached subpath and extend")
                        print("Warning: path has duplicates right before storing:", path)
                    # path.extend(path_cache[path_key])  # Extend path with cached successful route
                    successful_routes += 1
                    path_cache[(node, destination)] = path  # Cache the full path from the original node
                    break  # If path is known to succeed, stop processing

                # Check if direct connection to destination is possible within communication range
                distance_to_GS = all_results[current]['distance_to_GS']
                if distance_to_GS <= a2g_comm_range:
                    path.append(destination)  # Directly add the destination since it's within range
                    successful_routes += 1
                    path_cache[(node, destination)] = path  # Cache the full path from the original node

                    # Cache the full path for each node in the path
                    for i in range(len(path) - 1):
                        subpath_key = (path[i], destination)
                        if subpath_key not in path_cache:
                            path_cache[subpath_key] = path[i + 1:]  # Cache the subpath from node i to destination
                    break

                next_hop = find_next_hop(current, all_results, a2a_comm_range, k)

                if next_hop is None or next_hop in path:
                    failed_path_cache[path_key] = True  # Mark this as a failed route
                    for failed_node in path:
                        failed_path_cache[(failed_node, destination)] = True  # Mark all nodes in path as failed
                    break

                path.append(next_hop)
                current = next_hop

            # If we reached the destination through greedy routing (not directly via A2G), record the stretch factor
            if current == destination:
                successful_routes += 1

                # Cache the full path for each node in the path
                for i in range(len(path) - 1):
                    subpath_key = (path[i], destination)
                    if subpath_key not in path_cache:
                        path_cache[subpath_key] = path[i + 1:]  # Cache the subpath from node i to destination
    return successful_routes

def perform_centrality_based_geographic_routing(G, all_results, all_results_subset, a2a_comm_range, a2g_comm_range, destination, find_next_hop, m):
    """
    Performs centrality-based geographic routing by considering a subset of nodes.
    This function uses enhanced caching to improve efficiency and calculates probabilities of routing advances.

    Parameters:
    ----------
    G : networkx.Graph
        The graph over which nodes are routed.
    all_results_subset : dict
        Contains node details for subsets.
    a2a_comm_range : float
        The air-to-air communication range.
    a2g_comm_range : float
        The air-to-ground communication range.
    destination : int
        The identifier of the destination node.
    find_next_hop : function
        A callable that determines the best next hop based on betweenness centrality.
    m : int
        The subset size to consider for routing decisions.

    Returns:
    -------
    int:
        The number of successful routes completed.
    """
    successful_routes = 0
    path_cache = {}  # Cache for successful paths
    failed_path_cache = {}  # Cache for paths known to fail
    num_nodes = G.number_of_nodes() - 1

    # Compute hop counts from each node to the destination using Dijkstra
    hop_counts_to_dest = nx.single_source_shortest_path_length(G, destination)
    
    for node in G.nodes:
        if node != destination:
            try:
                # Find all Dijkstra paths from node to destination
                all_dijkstra_paths = list(nx.all_shortest_paths(G, node, destination))
                dijkstra_next_hops = list(set(path[1] for path in all_dijkstra_paths))
                dijkstra_path_length = len(all_dijkstra_paths[0]) - 1  # Hop count in Dijkstra path

            except nx.NetworkXNoPath:
                dijkstra_next_hops = None
                dijkstra_path_length = None
                geographic_next_hop = None
                failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
                continue  # Move to the next node since no path exists

            current = node
            path = [current]  # Initialize path with the current node

            # Compute the path if not in cache
            while current != destination:
                path_key = (current, destination)

                # Check failed path cache first
                if path_key in failed_path_cache:
                    failed_path_cache[(node, destination)] = True  # Add original node to failed path cache
                    break  # If intermediate path is known to fail, stop processing

                # Attempt to use the successful path cache
                if path_key in path_cache:
                    cached_subpath = path_cache[path_key]
                    if path and cached_subpath and path[-1] == cached_subpath[0]:
                        path.extend(cached_subpath[1:])
                    else:
                        path.extend(cached_subpath)
                    if len(path) != len(set(path)):
                        print("(A) When you find a cached subpath and extend")
                        print("Warning: path has duplicates right before storing:", path)
                    # path.extend(path_cache[path_key])  # Extend path with cached successful route
                    successful_routes += 1
                    path_cache[(node, destination)] = path  # Cache the full path from the original node
                    break  # If path is known to succeed, stop processing

                # Check if direct connection to destination is possible within communication range
                distance_to_GS = all_results_subset[m][current]['distance_to_GS']
                if distance_to_GS <= a2g_comm_range:
                    path.append(destination)  # Directly add the destination since it's within range
                    successful_routes += 1
                    path_cache[(node, destination)] = path  # Cache the full path from the original node

                    # Cache the full path for each node in the path
                    for i in range(len(path) - 1):
                        subpath_key = (path[i], destination)
                        if subpath_key not in path_cache:
                            path_cache[subpath_key] = path[i + 1:]  # Cache the subpath from node i to destination
                    break

                # Call the next-hop selection function with the subset size `m`
                next_hop = find_next_hop(current, all_results, all_results_subset, m)

                if next_hop is None or next_hop in path:
                    failed_path_cache[path_key] = True  # Mark this as a failed route
                    for failed_node in path:
                        failed_path_cache[(failed_node, destination)] = True  # Mark all nodes in path as failed
                    break

                path.append(next_hop)
                current = next_hop

            # If we reached the destination through greedy routing (not directly via A2G), record the stretch factor
            if current == destination:
                successful_routes += 1

                # Cache the full path for each node in the path
                for i in range(len(path) - 1):
                    subpath_key = (path[i], destination)
                    if subpath_key not in path_cache:
                        path_cache[subpath_key] = path[i + 1:]  # Cache the subpath from node i to destination

    return successful_routes

def run_simulation(args):
    # Unpack all necessary parameters
    num_nodes, rep, k = args

    # Set the seed for reproducibility
    np.random.seed(rep)

    # Settings and parameters
    area_length, area_side = 1250, 800
    a2a_comm_range = 100  # A2A Communication range in km
    a2g_comm_range = 370.4  # A2G Communication range in km
    GS_id = 501
    GS_pos = (1150, 400)
    m = 8  # Subset size for specific scenarios

    # Create the network and process all nodes
    G, positions = create_network(area_length, area_side, num_nodes, a2a_comm_range, a2g_comm_range, GS_id, GS_pos)
    all_node_details = process_all_nodes_with_betweenness_and_distance(G, positions, max_radius=k, GS_id=GS_id)
    all_node_details_subset = process_all_nodes_with_betweenness_with_subset(G, positions, [m], a2a_comm_range, GS_id=GS_id)

    # Scenario 1: Success Rate using Dijkstra's algorithm
    path_exists = 0
    for node in G.nodes:
        if node != GS_id and nx.has_path(G, node, GS_id):
            path_exists += 1
    success_rate_dijkstra = path_exists / num_nodes  # Exclude destination from source count

    # Scenario 2: Greedy-1, Greedy-2, Greedy-3
    greedy_results = {}
    for k in [1, 2, 3]:
        success_rate = perform_geographic_routing(
            G, all_node_details, a2a_comm_range, a2g_comm_range, GS_id, best_advance_within_k_hops, k
        )
        greedy_results[k] = {
            'success_rate': success_rate / path_exists if path_exists > 0 else 0
        }
        
    # Scenario 3: BCGR
    success_rate_bcgr = perform_geographic_routing(
            G, all_node_details, a2a_comm_range, a2g_comm_range, GS_id, BCGR, 1
    )
    success_rate_bcgr /= path_exists if path_exists > 0 else 0

    # Scenario 4: Greedy advance with betweenness using m=8
    success_rate_sbcr = perform_centrality_based_geographic_routing(
        G, all_node_details, all_node_details_subset, a2a_comm_range, a2g_comm_range, GS_id, subset_betweenness_centrality_routing, m
    )
    success_rate_sbcr /= path_exists if path_exists > 0 else 0

    return [
        num_nodes, rep,
        success_rate_dijkstra,
        greedy_results,
        success_rate_bcgr, 
        success_rate_sbcr, 
    ]


if __name__ == "__main__":
    # Settings and parameters
    a2a_comm_range = 100  # A2A Communication range in km
    # Number of repetitions for the simulation.
    # This should be set to 2000 for comprehensive analysis.
    # However, due to runtime limitations in Code Ocean,
    # we reduce the number of repetitions to 50 to ensure the code completes execution within allowed time frames.
    repititions = 50
    k = 3   # Can be changed to any desired value for k-hop
    num_node_values = np.arange(50, 501, 50)
    max_num_nodes = 500
    equipage_fraction_values = np.divide(num_node_values, max_num_nodes)

    # Specify the output directory and filename
    output_directory = '../results/AnalyzeCentralityBasedRouting_Submitted'
    output_filename = "AnalyzeCentralityBasedRouting.csv"
    full_path = os.path.join(output_directory, output_filename)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    parameters = list(product(num_node_values, range(repititions), [k]))
    try:
        # Attempt to use the SLURM environment variable to set the number of CPUs
        ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        print("Running under SLURM management. CPUs allocated:", ncpus)
    except KeyError:
        # Fallback to local CPU count if the SLURM environment variable is not found
        ncpus = cpu_count()
        print("Running on local machine. CPUs available:", ncpus)
        print(f"cpus_per_task: {ncpus}")  # output is coherent with my slurm script

    with Pool(ncpus) as pool:
            results = []
            print(f"cpus_per_task: {ncpus}")
            for result in pool.imap(run_simulation, parameters):
                results.append(result)
                (
                    num_nodes, rep,
                    success_rate_dijkstra,
                    greedy_results,
                    success_rate_bcgr,
                    success_rate_sbcr
                ) = result
                print(f"NumNodes: {num_nodes}, Rep: {rep}, Dijkstra: {success_rate_dijkstra:.3f}, "
                    f"Greedy-1: ({greedy_results[1]['success_rate']:.3f}), "
                    f"Greedy-2: ({greedy_results[2]['success_rate']:.3f}), "
                    f"Greedy-3: ({greedy_results[3]['success_rate']:.3f}), "
                    f"BCGR: ({success_rate_bcgr:.3f}), "
                    f"S_BCR: ({success_rate_sbcr:.3f})")

    # Initialize a structure to store the aggregated results for Dijkstra, Greedy-k, BCGR, Betweenness, and Degree
    aggregated_results = {
        num_nodes: {rep: {
            'Dijkstra': None,  # Stores the success rate for Dijkstra
            **{f'Greedy-{i}': {'success_rate': None} for i in range(1, 4)},
            'BCGR': {'success_rate': None},
            'S-BCR': {'success_rate': None}
        } for rep in range(repititions)}
        for num_nodes in num_node_values
    }

    # Populate this structure with results
    for result in results:
        (
            num_nodes, rep,
            success_rate_dijkstra,
            greedy_results,
            success_rate_bcgr,
            success_rate_sbcr, 
        ) = result
        
        # Populate Dijkstra results
        aggregated_results[num_nodes][rep]['Dijkstra'] = success_rate_dijkstra

        # Populate Greedy-k results
        for i in range(1, 4):
            aggregated_results[num_nodes][rep][f'Greedy-{i}'] = {
                'success_rate': greedy_results[i]['success_rate']
            }

        # Populate BCGR results
        aggregated_results[num_nodes][rep]['BCGR'] = {
            'success_rate': success_rate_bcgr
        }

        # Populate Betweenness results
        aggregated_results[num_nodes][rep]['S-BCR'] = {
            'success_rate': success_rate_sbcr
        }

    # Define the fieldnames (column headers) for the CSV
    headers = [
        'NumNodes', 'Avg_Dijkstra', 'MoE_Dijkstra',
        *[f'Avg_Greedy-{i}' for i in range(1, 4)],
        *[f'MoE_Greedy-{i}' for i in range(1, 4)],
        'Avg_BCGR', 'MoE_BCGR',
        'Avg_SBCR', 'MoE_SBCR'
    ]

    # Open the CSV file for writing
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()  # Write the header row

        # Write data rows
        for num_nodes in num_node_values:
            # Collect results for each scenario into lists
            all_dijkstra = [aggregated_results[num_nodes][rep]['Dijkstra'] for rep in range(repititions)]
            all_greedy = {f'Greedy-{i}': [aggregated_results[num_nodes][rep][f'Greedy-{i}']['success_rate'] for rep in range(repititions)] for i in range(1, 4)}


            all_bcgr = [aggregated_results[num_nodes][rep]['BCGR']['success_rate'] for rep in range(repititions)]

            all_sbcr = [aggregated_results[num_nodes][rep]['S-BCR']['success_rate'] for rep in range(repititions)]

            # Calculate means and MoEs for Dijkstra, Greedy-k, and GSR-Dresults
            mean_dijkstra, _, moe_dijkstra = confidence_interval_init(all_dijkstra)
            means_greedy = {}
            moes_greedy = {}
            for i in range(1, 4):
                means_greedy[f'Greedy-{i}'], _, moes_greedy[f'Greedy-{i}'] = confidence_interval_init(all_greedy[f'Greedy-{i}'])

            mean_bcgr, _, moe_bcgr = confidence_interval_init(all_bcgr)
            mean_sbcr, _, moe_sbcr = confidence_interval_init(all_sbcr)

            # Write the data row
            writer.writerow({
                'NumNodes': num_nodes,
                'Avg_Dijkstra': mean_dijkstra, 'MoE_Dijkstra': moe_dijkstra,
                **{f'Avg_Greedy-{i}': means_greedy[f'Greedy-{i}'] for i in range(1, 4)},
                **{f'MoE_Greedy-{i}': moes_greedy[f'Greedy-{i}'] for i in range(1, 4)},
                'Avg_BCGR': mean_bcgr, 'MoE_BCGR': moe_bcgr,
                'Avg_SBCR': mean_sbcr, 'MoE_SBCR': moe_sbcr
            })

    print(f"Saved final averages to {full_path}")

    # Initialize lists to hold data for plotting
    num_nodes = []
    final_mean_dijkstra, final_moe_dijkstra = [], []
    final_means_greedy = {f'Greedy-{i}': [] for i in range(1, 4)}
    final_moes_greedy = {f'Greedy-{i}': [] for i in range(1, 4)}

    final_mean_bcgr, final_moe_bcgr = [], []

    final_mean_sbcr, final_moe_sbcr = [], []

    # Read data from CSV
    with open(full_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            num_nodes.append(float(row['NumNodes']))
            final_mean_dijkstra.append(float(row['Avg_Dijkstra']))
            final_moe_dijkstra.append(float(row['MoE_Dijkstra']))
            
            for i in range(1, 4):
                final_means_greedy[f'Greedy-{i}'].append(float(row[f'Avg_Greedy-{i}']))
                final_moes_greedy[f'Greedy-{i}'].append(float(row[f'MoE_Greedy-{i}']))

            final_mean_bcgr.append(float(row['Avg_BCGR']))
            final_moe_bcgr.append(float(row['MoE_BCGR']))

            final_mean_sbcr.append(float(row['Avg_SBCR']))
            final_moe_sbcr.append(float(row['MoE_SBCR']))


    # Ensure the arrays are sorted by node counts
    sorted_indices = np.argsort(num_nodes)
    num_nodes = np.array(num_nodes)[sorted_indices]
    final_mean_dijkstra = np.array(final_mean_dijkstra)[sorted_indices]
    final_moe_dijkstra = np.array(final_moe_dijkstra)[sorted_indices]

    for i in range(1, 4):
        final_means_greedy[f'Greedy-{i}'] = np.array(final_means_greedy[f'Greedy-{i}'])[sorted_indices]
        final_moes_greedy[f'Greedy-{i}'] = np.array(final_moes_greedy[f'Greedy-{i}'])[sorted_indices]

    final_mean_bcgr = np.array(final_mean_bcgr)[sorted_indices]
    final_moe_bcgr = np.array(final_moe_bcgr)[sorted_indices]

    final_mean_sbcr = np.array(final_mean_sbcr)[sorted_indices]
    final_moe_sbcr = np.array(final_moe_sbcr)[sorted_indices]

    print(f"Results processed and saved to {full_path}.")

    # Plotting results
    plt.figure(figsize=(12, 9))
    capsize = 4
    markersize = 5
    lw = 3
    colors_map = [plt.cm.jet(x) for x in np.linspace(0, 1, 8)]  # Generate colors for each plot line
    colors = ["#FFBB6F", "#A00000", "#7A7A7A", "#298C8C", "#006400"] # Gold, Red, Gray, Teal and Darkgreen
    linestyles = ['dotted', 'dashed', 'solid', 'dashdot']  # Define different line styles

    # Plot 1: Success Rates for Dijkstra, Greedy-k, BCGR, Betweenness, and Degree
    for i in range(1, 4):
        plt.errorbar(
            equipage_fraction_values,
            final_means_greedy[f'Greedy-{i}'],
            yerr=final_moes_greedy[f'Greedy-{i}'],
            label=fr'Greedy-{i}',
            linestyle=linestyles[i-1 % len(linestyles)],
            fmt='o',  # Cycle through the markers
            markersize=markersize,
            markeredgecolor='black',
            lw=lw,
            capsize=capsize,
            color=colors[2]
        )

    plt.errorbar(
        equipage_fraction_values,
        final_mean_bcgr,
        yerr=final_moe_bcgr,
        label='BCGR',
        linestyle='solid',
        fmt='o',
        markersize=markersize,
        markeredgecolor='black',
        lw=lw,
        capsize=capsize,
        color=colors[1]
    )

    plt.errorbar(
        equipage_fraction_values,
        final_mean_sbcr,
        yerr=final_moe_sbcr,
        label='S-BCR',
        linestyle='solid',
        fmt='o',
        markersize=markersize,
        markeredgecolor='black',
        lw=lw,
        capsize=capsize,
        color=colors[4]
    )

    # annotate scenario B
    plt.axvline(0.4, color='k', linestyle="--", linewidth=2, alpha=0.8)
    # Annotate with a white box and connecting line
    plt.annotate(
        f"Scenario B",
        xy=(0.4, 0.97),  # Place annotation at the x-value of the mean, starting at the bottom (y=0)
        xytext=(0.4, 0.97),  # Adjust the y-value for positioning the text
        textcoords="data",  # Use data coordinates for placing the text
        horizontalalignment="center",
        verticalalignment="center",
        # fontsize=20,
        bbox=dict(facecolor="white", edgecolor='k', boxstyle="round,pad=0.3", linewidth = lw),
        # arrowprops=dict(facecolor=colors_map[radius], arrowstyle="->", connectionstyle="arc3,rad=0.5")
    )

    # annotate scenario A
    plt.axvline(0.8, color='k', linestyle="--", linewidth=2, alpha=0.8)
    # Annotate with a white box and connecting line
    plt.annotate(
        f"Scenario A",
        xy=(0.8, 0.84),  # Place annotation at the x-value of the mean, starting at the bottom (y=0)
        xytext=(0.8, 0.84),  # Adjust the y-value for positioning the text
        textcoords="data",  # Use data coordinates for placing the text
        horizontalalignment="center",
        verticalalignment="center",
        # fontsize=20,
        bbox=dict(facecolor="white", edgecolor='k', boxstyle="round,pad=0.3", linewidth = lw),
        # arrowprops=dict(facecolor=colors_map[radius], arrowstyle="->", connectionstyle="arc3,rad=0.5")
    )

    plt.ylim([0.68, 1.015])

    # Set axis labels and tick formatting
    plt.xlabel('Equipage Fraction')
    plt.ylabel('Success Ratio')

    # Configure y-axis locators
    plt.gca().yaxis.set_major_locator(ticker.AutoLocator())
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Configure x-axis locators
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # Set up grid
    plt.gca().xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    plt.gca().xaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    plt.gca().yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    plt.gca().yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)

    # Add a legend
    plt.legend(loc='best', ncol=2)
    plt.tight_layout()

    # Save the plots
    plt.savefig(os.path.join(output_directory, f"success_rate_analysis_r={a2a_comm_range}_k={k}.pdf"), format='pdf', bbox_inches="tight")

    # Display the plot
    plt.show()
    plt.close()

