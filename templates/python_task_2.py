import pandas as pd
import numpy as np
import networkx as nx
from datetime import time

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    # Create a directed graph to represent routes and distances
    graph = nx.DiGraph()

    # Iterate through the DataFrame to build the graph
    for index, row in df.iterrows():
        # Add bidirectional edges with cumulative distances
        graph.add_edge(row['id_start'], row['id_end'], distance=row['distance'])
        graph.add_edge(row['id_end'], row['id_start'], distance=row['distance'])

    # Calculate shortest path lengths to get the distances
    distance_matrix = np.zeros((len(graph.nodes), len(graph.nodes)))
    for i, start_node in enumerate(graph.nodes):
        for j, end_node in enumerate(graph.nodes):
            if i != j:
                distance = nx.shortest_path_length(graph, start_node, end_node, weight='distance')
                distance_matrix[i][j] = distance

    # Convert the distance matrix to a DataFrame
    distance_df = pd.DataFrame(distance_matrix, index=graph.nodes, columns=graph.nodes)

    return distance_df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    # Extract lower triangular part (excluding diagonal) of the distance matrix
    lower_triangular = np.tril(df.values, k=-1)
    
    # Get row and column indices for lower triangular elements
    rows, cols = np.where(lower_triangular > 0)
    
    # Create DataFrame with 'id_start', 'id_end', and 'distance' columns
    unrolled_df = pd.DataFrame({
        'id_start': np.round(df.index[cols]).astype(int),
        'id_end': np.round(df.columns[rows]).astype(int),
        'distance': lower_triangular[lower_triangular > 0]
    })
    
    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    # Calculate average distance for the reference ID
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    # Calculate the threshold range within 10% of the reference ID's average distance
    lower_threshold = reference_avg_distance * 0.9
    upper_threshold = reference_avg_distance * 1.1

    # Filter IDs within the specified threshold range
    filtered_ids = df[(df['id_start'] != reference_id) & 
                      (df['distance'] >= lower_threshold) & 
                      (df['distance'] <= upper_threshold)]
    
    # Get unique IDs within the threshold range and sort them
    result = filtered_ids['id_start'].unique()
    result.sort()

    return pd.DataFrame({'id_start': result})


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates by multiplying distance with rate coefficients for each vehicle type
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    # Define time ranges and their respective discount factors
    weekday_time_ranges = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]
    weekend_time_factor = 0.7

    # Create new columns for start_day, end_day, start_time, and end_time
    df['start_day'] = pd.to_datetime(df['id_start']).dt.day_name()
    df['end_day'] = pd.to_datetime(df['id_end']).dt.day_name()
    df['start_time'] = pd.to_datetime(df['id_start']).dt.time
    df['end_time'] = pd.to_datetime(df['id_end']).dt.time

    # Iterate through rows and modify vehicle columns based on time intervals
    for index, row in df.iterrows():
        start_time = df.at[index, 'start_time']
        end_time = df.at[index, 'end_time']
        start_day = df.at[index, 'start_day']
        end_day = df.at[index, 'end_day']

        if start_day in ['Saturday', 'Sunday']:
            discount_factor = weekend_time_factor
        else:
            for start, end, factor in weekday_time_ranges:
                if start <= start_time <= end and start <= end_time <= end:
                    discount_factor = factor
                    break
        
        # Apply discount factor to vehicle columns
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            df.at[index, vehicle] *= discount_factor

    return df


dataset_3 = pd.read_csv('/home/abhishek/Code/MapUp-Data-Assessment-F/datasets/dataset-3.csv')
car_matrix = calculate_distance_matrix(dataset_3)
car_matrix1 = unroll_distance_matrix(car_matrix)
car_matrix2 = find_ids_within_ten_percentage_threshold(car_matrix1, 1001400)
car_matrix3 = calculate_toll_rate(car_matrix1)
car_matrix4 = calculate_time_based_toll_rates(car_matrix2)
# Print the car matrix.
print(car_matrix4)