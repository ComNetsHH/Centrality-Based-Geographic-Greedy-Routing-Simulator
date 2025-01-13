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
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import pandas as pd
from scipy.stats import binned_statistic_2d
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def calculate_pairwise_distances(points, selected_points):
    # Convert points and selected_points to NumPy arrays
    points_array = np.array([p['pos'] for p in points])
    selected_points_array = np.array([sp['pos'] for sp in selected_points])

    # Compute pairwise distances
    return euclidean_distances(points_array, selected_points_array)

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def select_EFFT_neighbors(neighbor_attr, points, m, R=100, threshold = None):
    # """Select up to k unique nodes from a set of neighbors in graph G using standard Farthest-First Traversal."""
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

def apply_efft_algorithm(G, positions, m_values, comm_range):
    """
    Apply EFFT logic to determine subsets and neighbors for 1-hop, 2-hop, and 3-hop neighborhoods.
    """
    result_subset = {m: {node: {"S1": set(), "S2": set()} for node in G.nodes()} for m in m_values}
    result_neighbors = {m: {node: {"1-hop": set(), "2-hop": set(), "3-hop": set()} for node in G.nodes()} for m in m_values}
    # Step 1: Find 1-hop neighbors and calculate S^1_i for each node
    for m in m_values:
        for node in G.nodes():
            neighbors = set(nx.neighbors(G, node))
            result_neighbors[m][node]["1-hop"] = {(neighbor, neighbor) for neighbor in neighbors}
            neighbor_attrs = [{"id": neighbor, "pos": positions[neighbor]} for neighbor in neighbors]
            subset_S1 = set(select_EFFT_neighbors({"pos": positions[node]}, neighbor_attrs, m, comm_range))
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
                subset_S2 = set(select_EFFT_neighbors({"pos": positions[node]}, two_hop_attrs, remaining_m, comm_range, threshold = 0))
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

def calculate_betweenness(G, k, GS_id=501):
    # Remove GS_id from the graph
    if GS_id in G:
        G.remove_node(GS_id)

    # Calculate global betweenness centrality
    global_betweenness = nx.betweenness_centrality(G, normalized=False)

    # Dictionary to store local betweenness centrality for each hop level
    local_betweenness = {}
    neighbors_by_hop = {n: {i: set() for i in range(1, k + 1)} for n in G.nodes()}

    for n in G.nodes():
        if n == GS_id:
            continue  # Skip processing for the destination node

        # Compute shortest paths and their lengths from node n up to k hops
        sp_paths = nx.single_source_shortest_path(G, n, cutoff=k)
        nodes_by_hop = {i: set() for i in range(1, k+1)}
        
        # Categorize nodes by hop count from n
        for target, path in sp_paths.items():
            hop_count = len(path) - 1
            if hop_count > 0:
                nodes_by_hop[hop_count].add(target)

        neighbors_by_hop[n] = nodes_by_hop

        # Compute betweenness centrality for each subgraph formed by nodes up to each hop level
        accumulated_nodes = set()
        # Explicitly add the current node to the accumulated nodes
        accumulated_nodes.update([n])
        for hop in range(1, k+1):
            accumulated_nodes.update(nodes_by_hop[hop])            
            subgraph = G.subgraph(accumulated_nodes)
            
            local_betweenness_for_n = nx.betweenness_centrality(subgraph, normalized=False)

            # Store the betweenness centrality of node n at this hop level
            local_betweenness[(n, hop)] = local_betweenness_for_n.get(n, 0)
    return global_betweenness, local_betweenness, neighbors_by_hop

def calculate_betweenness_subset(G, final_neighbors, m_values):
    """
    Calculate local betweenness centrality for subsets defined in final_neighbors for each m-value.

    Parameters:
    ----------
    G : graph
        A NetworkX Graph or DiGraph.

    final_neighbors : dict
        A dictionary of neighbors by hop level for each m-value and node.

    m_values : list
        A list of m-values to calculate subsets for.

    Returns:
    -------
    dict: A dictionary where keys are (n, m) and values are the local betweenness centrality for the node n and subset size m.
    """
    local_betweenness_subset = {}

    for m in m_values:
        for node in G.nodes():
            # Collect nodes from final_neighbors for all hops
            subset_nodes = {node}  # Include the current node itself
            for hop in ["1-hop", "2-hop", "3-hop"]:
                subset_nodes.update(neighbor for neighbor, _ in final_neighbors[m][node][hop])

            # Create the subgraph induced by the subset nodes
            subgraph = G.subgraph(subset_nodes)

            # Calculate betweenness centrality for the subgraph
            betweenness = nx.betweenness_centrality(subgraph, normalized=False)

            # Store the local betweenness for the current node
            local_betweenness_subset[(node, m)] = betweenness.get(node, 0)

    return local_betweenness_subset

def calculate_spearman_correlation(global_betweenness, local_betweenness, neighbors_by_hop, k):
    spearman_results = {hop: [] for hop in range(1, k + 1)}
    for node, neighbors in neighbors_by_hop.items():
        for hop in range(1, k + 1):
            # Ensure the current node and its neighbors at this hop are included
            current_hop_neighbors = neighbors[hop].union({node})  # Add the current node to the neighbors
            
            if not current_hop_neighbors:
                continue

            # Create a DataFrame for neighbors at the given hop level
            df = pd.DataFrame({
                'Neighbor': list(current_hop_neighbors),
                'GlobalBetweenness': [global_betweenness.get(neigh, 0) for neigh in current_hop_neighbors],
                'LocalBetweenness': [local_betweenness.get((neigh, hop), 0) for neigh in current_hop_neighbors]
            })
            
            # Rank the columns
            df['GlobalRank'] = df['GlobalBetweenness'].rank()
            df['LocalRank'] = df['LocalBetweenness'].rank()

            # Skip calculation if input is constant
            if df['GlobalRank'].nunique() > 1 and df['LocalRank'].nunique() > 1:
                corr, _ = spearmanr(df['GlobalRank'], df['LocalRank'])
                spearman_results[hop].append(corr)

    # Aggregate results (average correlation for each hop level) -> handle None values before calculating the average
    averaged_results = {
        hop: np.mean([v for v in values if v is not None]) if values else None 
        for hop, values in spearman_results.items()
    }
    return averaged_results

def run_simulation(args):
    num_nodes, rep, k = args

    # Set the seed for reproducibility
    np.random.seed(rep)

    # Settings and parameters
    area_length, area_side = 1250, 800
    a2a_comm_range = 100  # A2A Communication range in km
    a2g_comm_range = 370.4  # A2G Communication range in km
    GS_id = 501
    GS_pos = (1150, 400)

    # Create the network and process all nodes
    G, positions = create_network(area_length, area_side, num_nodes, a2a_comm_range, a2g_comm_range, GS_id, GS_pos)
    global_betweenness, local_betweenness, neighbors_by_hop = calculate_betweenness(G, k)

    # Generate EFFT subsets and calculate local betweenness for subsets
    final_neighbors = apply_efft_algorithm(G, positions, m_values, a2a_comm_range)
    local_betweenness_subset = calculate_betweenness_subset(G, final_neighbors, m_values)

    # Initialize correlation dictionaries
    pearson_correlations = {}
    spearman_correlations = {}
    pearson_correlations_subset = {m: {} for m in m_values}
    spearman_correlations_subset = {m: {} for m in m_values}
    
    # Calculate Spearman’s rank correlation
    # spearman_correlations = calculate_spearman_correlation(global_betweenness, local_betweenness, neighbors_by_hop, k)
    for radius in range(1, k + 1):
        local_vals = [local_betweenness[(n, radius)] for n in G.nodes() if n != GS_id]
        global_vals = [global_betweenness[n] for n in G.nodes() if n != GS_id]
        pearson_correlation = pearsonr(local_vals, global_vals)[0]
        spearman_correlation = spearmanr(local_vals, global_vals)[0]
        pearson_correlations[radius] = pearson_correlation
        spearman_correlations[radius] = spearman_correlation

    # Calculate Pearson and Spearman correlations for subsets
    for m in m_values:
        for node in G.nodes():
            if node == GS_id:
                continue
            # Collect subset values for the current m-value
            subset_vals = [local_betweenness_subset[(node, m)] for node in G.nodes() if node != GS_id]
            global_vals = [global_betweenness[n] for n in G.nodes() if n != GS_id]
            # Calculate correlations
            pearson_correlation_subset = pearsonr(subset_vals, global_vals)[0]
            spearman_correlation_subset = spearmanr(subset_vals, global_vals)[0]
            pearson_correlations_subset[m] = pearson_correlation_subset
            spearman_correlations_subset[m] = spearman_correlation_subset

    return {
        "num_nodes": num_nodes,
        "repetition": rep,
        "pearson_correlations": pearson_correlations,
        "spearman_correlations": spearman_correlations,
        "pearson_correlations_subset": pearson_correlations_subset,
        "spearman_correlations_subset": spearman_correlations_subset,
        "local_betweenness": local_betweenness,
        "local_betweenness_subset": local_betweenness_subset,
        "global_betweenness": global_betweenness,
    }

if __name__ == "__main__":
    # Define simulation settings
    # Number of repetitions for the simulation.
    # This should be set to 2000 for comprehensive analysis.
    # However, due to runtime limitations in Code Ocean,
    # we reduce the number of repetitions to 50 to ensure the code completes execution within allowed time frames.
    repititions = 50
    num_node_values = np.arange(50, 501, 50)
    k = 3  # maximum number of hops to calculate betweenness
    m_values = [4, 6, 8]  # Subset sizes
    a2a_comm_range = 100
    max_num_nodes = 500
    equipage_fraction_values = np.divide(num_node_values, max_num_nodes)
    m_plotting = 8


    selectedNumNodes = 200
    random_sample_scatter_size = 10000

    # Define your output directory and file
    output_directory = '../results/AnalyzeEgoBetweenness_Submitted'
    output_filename = 'AnalyzeEgoBetweennessCorrelation.csv'
    full_path = os.path.join(output_directory, output_filename)
    betweenness_csv_path = os.path.join(output_directory, "betweenness_data.csv")
    subset_betweenness_csv_path = os.path.join(output_directory, "subset_betweenness_data.csv")

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Set up simulation parameters
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

    print(f"cpus_per_task: {ncpus}")  # output is coherent with my slurm script
    # Use a multiprocessing pool to execute simulations
    with Pool(ncpus) as pool:
        results = []
        for result in pool.imap(run_simulation, parameters):
            results.append(result)
            print(
                f"Result received: NumNodes={result['num_nodes']}, "
                f"Repetition={result['repetition']}, "
                f"Pearson Correlations={result['pearson_correlations']}, "
                f"Spearman Correlations={result['spearman_correlations']}, "
                f"Pearson Correlations Subset={result['pearson_correlations_subset']}, "
                f"Spearman Correlations Subset={result['spearman_correlations_subset']}"
            )

    # Initialize a DataFrame for all repetitions
    betweenness_df = pd.DataFrame(columns=["NumNodes", "Rep", "Radius", "Node", "LocalBetweenness", "GlobalBetweenness"])
    betweenness_rows = []
    subset_betweenness_rows = []

    # Initialize aggregated_results with the required structure
    aggregated_results = {
        num_nodes: {
            rep: {
                "pearson": {i: [] for i in range(1, k+1)},  # Pearson correlations for each radius
                "spearman": {i: [] for i in range(1, k+1)},  # Spearman correlations for each radius
                "pearson_subset": {m: [] for m in m_values},  # Pearson correlations for each subset size
                "spearman_subset": {m: [] for m in m_values}  # Spearman correlations for each subset size
            }
            for rep in range(repetitions)
        }
        for num_nodes in num_node_values
    }


    for result in results:
        num_nodes = result["num_nodes"]
        rep = result["repetition"]
        pearson_correlations = result["pearson_correlations"]
        spearman_correlations = result["spearman_correlations"]
        local_betweenness = result["local_betweenness"]
        global_betweenness = result["global_betweenness"]
        local_betweenness_subset = result["local_betweenness_subset"]
        pearson_correlations_subset = result["pearson_correlations_subset"]
        spearman_correlations_subset = result["spearman_correlations_subset"]

        # Process correlations for each radius
        for radius in range(1, k+1):
            aggregated_results[num_nodes][rep]["pearson"][radius].append(pearson_correlations[radius])
            aggregated_results[num_nodes][rep]["spearman"][radius].append(spearman_correlations[radius])

        # Process subset correlations for each m-value
        for m in m_values:
            aggregated_results[num_nodes][rep]["pearson_subset"][m].append(pearson_correlations_subset[m])
            aggregated_results[num_nodes][rep]["spearman_subset"][m].append(spearman_correlations_subset[m])

        # Prepare rows for the betweenness DataFrame
        for (node, radius), local_value in local_betweenness.items():
            betweenness_rows.append({
                "NumNodes": num_nodes,
                "Rep": rep,
                "Radius": radius,
                "Node": node,
                "LocalBetweenness": local_value,
                "GlobalBetweenness": global_betweenness.get(node, 0)
            })

        # Prepare rows for the subset betweenness DataFrame
        for (node, m), local_value in local_betweenness_subset.items():
            subset_betweenness_rows.append({
                "NumNodes": num_nodes,
                "Repetition": rep,
                "SubsetSize": m,
                "Node": node,
                "LocalBetweennessSubset": local_value,
                "GlobalBetweenness": global_betweenness.get(node, 0)
            })



    # Create the DataFrame after accumulating all rows
    betweenness_df = pd.DataFrame(betweenness_rows)
    betweenness_df.to_csv(betweenness_csv_path, index=False)

    # Save subset betweenness data
    subset_betweenness_df = pd.DataFrame(subset_betweenness_rows)
    subset_betweenness_df.to_csv(subset_betweenness_csv_path, index=False)

    # Initialize headers for CSV
    headers = ['NumNodes']
    for i in range(1, k+1):
        headers.extend([f'{i}-Hop_Pearson_Avg', f'{i}-Hop_Pearson_MoE',
                        f'{i}-Hop_Spearman_Avg', f'{i}-Hop_Spearman_MoE'])

    # Extend headers for subset data
    for m in m_values:
        headers.extend([f'Pearson_Subset_Avg_m={m}', f'Pearson_Subset_MoE_m={m}',
                        f'Spearman_Subset_Avg_m={m}', f'Spearman_Subset_MoE_m={m}'])


    # Open the CSV file for writing
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        # Write aggregated results to CSV
        for num_nodes in num_node_values:
            row = {'NumNodes': num_nodes}
            for i in range(1, k+1):
                # all_correlations = [correlation for rep in range(repetitions) for correlation in aggregated_results[num_nodes][rep][i]]
                # mean_corr, _,  moe_corr = confidence_interval_init(all_correlations)
                pearson_correlations = [corr for rep in range(repetitions) for corr in aggregated_results[num_nodes][rep]["pearson"][i]]
                spearman_correlations = [corr for rep in range(repetitions) for corr in aggregated_results[num_nodes][rep]["spearman"][i]]
                pearson_mean, _, pearson_moe = confidence_interval_init(pearson_correlations)
                spearman_correlations = [val for val in spearman_correlations if val is not None]
                spearman_mean, _, spearman_moe = confidence_interval_init(spearman_correlations)

                # row[f'{i}-Hop_Avg'] = mean_corr
                # row[f'{i}-Hop_MoE'] = moe_corr
                row[f'{i}-Hop_Pearson_Avg'] = pearson_mean
                row[f'{i}-Hop_Pearson_MoE'] = pearson_moe
                row[f'{i}-Hop_Spearman_Avg'] = spearman_mean
                row[f'{i}-Hop_Spearman_MoE'] = spearman_moe
            
            # Add subset data for each m-value
            for m in m_values:
                pearson_subset_correlations = [corr for rep in range(repetitions) for corr in aggregated_results[num_nodes][rep]["pearson_subset"][m]]
                spearman_subset_correlations = [corr for rep in range(repetitions) for corr in aggregated_results[num_nodes][rep]["spearman_subset"][m]]
                pearson_subset_mean, _, pearson_subset_moe = confidence_interval_init(pearson_subset_correlations)
                spearman_subset_correlations = [val for val in spearman_subset_correlations if val is not None]
                spearman_subset_mean, _, spearman_subset_moe = confidence_interval_init(spearman_subset_correlations)

                row[f'Pearson_Subset_Avg_m={m}'] = pearson_subset_mean
                row[f'Pearson_Subset_MoE_m={m}'] = pearson_subset_moe
                row[f'Spearman_Subset_Avg_m={m}'] = spearman_subset_mean
                row[f'Spearman_Subset_MoE_m={m}'] = spearman_subset_moe

            writer.writerow(row)

    # Save the aggregated DataFrame

    # filtered_betweenness.to_csv(betweenness_csv_path, index=False)

    print(f"Saved final averages to {full_path} and {betweenness_csv_path}")

    # Initialize lists to hold data for plotting
    num_nodes = []
    # correlation_averages = {i: [] for i in range(1, k+1)}
    # correlation_moes = {i: [] for i in range(1, k+1)}
    pearson_averages = {i: [] for i in range(1, k+1)}
    pearson_moes = {i: [] for i in range(1, k+1)}
    spearman_averages = {i: [] for i in range(1, k+1)}
    spearman_moes = {i: [] for i in range(1, k+1)}
    pearson_subset_averages = {m: [] for m in m_values}
    pearson_subset_moes = {m: [] for m in m_values}
    spearman_subset_averages = {m: [] for m in m_values}
    spearman_subset_moes = {m: [] for m in m_values}


    # Read the CSV back into a DataFrame
    betweenness_df_loaded = pd.read_csv(betweenness_csv_path)
    subset_betweenness_df_loaded = pd.read_csv(subset_betweenness_csv_path)
    print("Betweenness data loaded from CSV.")

    # Read data from CSV
    with open(full_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            num_nodes.append(float(row['NumNodes']))
            for i in range(1, k+1):
                # correlation_averages[i].append(float(row[f'{i}-Hop_Avg']))
                # correlation_moes[i].append(float(row[f'{i}-Hop_MoE']))
                pearson_averages[i].append(float(row[f'{i}-Hop_Pearson_Avg']))
                pearson_moes[i].append(float(row[f'{i}-Hop_Pearson_MoE']))
                spearman_averages[i].append(float(row[f'{i}-Hop_Spearman_Avg']))
                spearman_moes[i].append(float(row[f'{i}-Hop_Spearman_MoE']))

            # Load subset data
            for m in m_values:
                pearson_subset_averages[m].append(float(row[f'Pearson_Subset_Avg_m={m}']))
                pearson_subset_moes[m].append(float(row[f'Pearson_Subset_MoE_m={m}']))
                spearman_subset_averages[m].append(float(row[f'Spearman_Subset_Avg_m={m}']))
                spearman_subset_moes[m].append(float(row[f'Spearman_Subset_MoE_m={m}']))

    # Ensure the arrays are sorted by node counts if not already
    sorted_indices = np.argsort(num_nodes)
    num_nodes = np.array(num_nodes)[sorted_indices]
    for i in range(1, k+1):
        # correlation_averages[i] = np.array(correlation_averages[i])[sorted_indices]
        # correlation_moes[i] = np.array(correlation_moes[i])[sorted_indices]
        pearson_averages[i] = np.array(pearson_averages[i])[sorted_indices]
        pearson_moes[i] = np.array(pearson_moes[i])[sorted_indices]
        spearman_averages[i] = np.array(spearman_averages[i])[sorted_indices]
        spearman_moes[i] = np.array(spearman_moes[i])[sorted_indices]
    for m in m_values:
        pearson_subset_averages[m] = np.array(pearson_subset_averages[m])[sorted_indices]
        pearson_subset_moes[m] = np.array(pearson_subset_moes[m])[sorted_indices]
        spearman_subset_averages[m] = np.array(spearman_subset_averages[m])[sorted_indices]
        spearman_subset_moes[m] = np.array(spearman_subset_moes[m])[sorted_indices]

    # Plotting results
    plt.figure(figsize=(12, 9))
    capsize = 4
    markersize = 5
    lw = 3

    colors = ["#FFBB6F", "#A00000", "#7A7A7A", "#298C8C", "#006400"] # Gold, Red, Gray, Teal and Darkgreen
   
    # markers = ['s', 'D', '*', 'x', 'P']  # Add more markers if needed
    markers = ['s', 'D', '*', 'x', 'P', 'o', '^']  # Added 'o' (circle) and '^' (triangle-up)
    # linestyles = ['solid', 'dashed', 'dotted', 'dashdot']  # Define different line styles
    linestyles = ['dotted', 'dashed', 'solid', 'dashdot']  # Define different line styles

    for i in range(1, k+1):
        plt.errorbar(
            equipage_fraction_values,
            spearman_averages[i],
            # yerr=spearman_moes[i],
            label=fr'{i}-Hop',
            # linestyle='solid',
            linestyle=linestyles[i-1 % len(linestyles)],  # Cycle through the linestyles
            fmt=markers[(i-1 % len(markers))],  # Cycle through the markers
            markersize=markersize,
            markeredgecolor='black',
            lw=lw,
            capsize=capsize,
            # color=colors_map[i-1]
            color=colors[-2]
        )

    # Plot the subset correlation results for subsets
    i = 0
    for m in m_values:
        plt.errorbar(
            equipage_fraction_values,
            spearman_subset_averages[m],
            label=fr'EFF (m={m})',
            linestyle=linestyles[i % len(linestyles)],  # Use a unique linestyle for subset data
            fmt=markers[i+3],  # Cycle through the markers
            markersize=markersize,
            markeredgecolor='black',
            lw=lw,
            capsize=capsize,
            color=colors[-3]
        )
        i += 1

    # annotate scenario B
    plt.axvline(0.4, color='k', linestyle="--", linewidth=2, alpha=0.8)
    # Annotate with a white box and connecting line
    plt.annotate(
        f"Scenario B",
        xy=(0.4, 0.65),  # Place annotation at the x-value of the mean, starting at the bottom (y=0)
        xytext=(0.4, 0.65),  # Adjust the y-value for positioning the text
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
        xy=(0.8, 0.65),  # Place annotation at the x-value of the mean, starting at the bottom (y=0)
        xytext=(0.8, 0.65),  # Adjust the y-value for positioning the text
        textcoords="data",  # Use data coordinates for placing the text
        horizontalalignment="center",
        verticalalignment="center",
        # fontsize=20,
        bbox=dict(facecolor="white", edgecolor='k', boxstyle="round,pad=0.3", linewidth = lw),
        # arrowprops=dict(facecolor=colors_map[radius], arrowstyle="->", connectionstyle="arc3,rad=0.5")
    )
    
    # Set axis labels and tick formatting
    plt.xlabel('Equipage Fraction')
    plt.ylabel('Spearman Correlation Coefficient')

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

    plt.ylim(0.3, 1)

    # Add legend with two columns
    plt.legend(loc='best', ncol=2)
    plt.tight_layout()

    # Save the plots
    plt.savefig(os.path.join(output_directory, f"analyze_spearman_correlation_rate_r={a2a_comm_range}_k={k}.pdf"), format='pdf', bbox_inches="tight")
    # plt.savefig(os.path.join(output_directory, f"analyze_spearman_correlation_rate_r={a2a_comm_range}_k={k}.png"), format='png', bbox_inches="tight")

    # Display the plot
    plt.show()
    plt.close()


    # Assume betweenness_df_loaded is already loaded with necessary data
    scatter_data = betweenness_df_loaded[betweenness_df_loaded["NumNodes"] == selectedNumNodes]

    cmaps_colors = ['Reds', 'Greens', 'Blues']

    # Define color map for different radii
    radii = scatter_data["Radius"].unique()
    fig, axes = plt.subplots(1, len(radii), figsize=(16, 6), sharey=False)

    for i, (ax, radius) in enumerate(zip(axes, radii)):
        # Filter data by radius
        radius_data = scatter_data[scatter_data["Radius"] == radius]
        
        # Compute the density of points using binned statistics
        statistic, x_edge, y_edge, _ = binned_statistic_2d(
            radius_data["GlobalBetweenness"],
            radius_data["LocalBetweenness"],
            None, 'count', bins=70)  # Adjust bin size as needed
        
        # Normalize the counts by the total number of points to show ratios
        total_points = np.sum(statistic)
        statistic_normalized = statistic / total_points if total_points != 0 else statistic
        
        # Calculate log of the counts (log(1 + count) to handle zeros)
        statistic_log = np.log1p(statistic)

        # Plot heatmap
        heatmap = ax.imshow(statistic_log.T, origin='lower', extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]], aspect='auto', cmap=cmaps_colors[i])
        ax.set_title(fr'{radius}-Hop')
        ax.set_xlabel("Global Betweenness")
        if ax == axes[0]:
            ax.set_ylabel("Local Betweenness")
        ax.grid(False)

            
    # Set layout
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(os.path.join(output_directory, f"analyze_betweeness_heatmap_r={a2a_comm_range}_n={selectedNumNodes}.pdf"), format='pdf', bbox_inches="tight")

    # Show the plot
    plt.close()

    # Assume betweenness_df_loaded is already loaded with necessary data
    scatter_data = betweenness_df_loaded[betweenness_df_loaded["NumNodes"] == selectedNumNodes]

    # Add GlobalRank and LocalRank columns to scatter_data directly for all reps and radii
    scatter_data["GlobalRank"] = scatter_data.groupby(["Rep", "Radius"])["GlobalBetweenness"].rank()
    scatter_data["LocalRank"] = scatter_data.groupby(["Rep", "Radius"])["LocalBetweenness"].rank()

    cmaps_colors = ['Reds', 'Greens', 'Blues']
   
    # Define color map for different radii
    radii = scatter_data["Radius"].unique()
    fig, axes = plt.subplots(1, len(radii), figsize=(16, 6), sharey=False)

    for i, (ax, radius) in enumerate(zip(axes, radii)):
        # Filter data by radius
        radius_data = scatter_data[scatter_data["Radius"] == radius]
        
        # Compute the density of points using binned statistics
        statistic, x_edge, y_edge, _ = binned_statistic_2d(
            radius_data["GlobalRank"],
            radius_data["LocalRank"],
            None, 'count', bins=70)  # Adjust bin size as needed
        
        # Calculate log of the counts (log(1 + count) to handle zeros)
        statistic_log = np.log1p(statistic)
        
        # Plot heatmap
        heatmap = ax.imshow(statistic_log.T, origin='lower', extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]], aspect='auto', cmap=cmaps_colors[i])
        ax.set_title(fr'{radius}-Hop')
        ax.set_xlabel("Global Betweenness Rank")
        if ax == axes[0]:
            ax.set_ylabel("Local Betweenness Rank")
        ax.grid(False)
            
    # Set layout
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(os.path.join(output_directory, f"analyze_betweeness_rank_heatmap_r={a2a_comm_range}_n={selectedNumNodes}.pdf"), format='pdf', bbox_inches="tight")

    # Show the plot
    plt.close()
