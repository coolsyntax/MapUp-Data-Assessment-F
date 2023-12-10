import pandas as pd
import numpy as np

def generate_car_matrix(df: pd.DataFrame)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    # Create a new DataFrame with the specified columns and index.
    # Pivot the DataFrame
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    
    # Set diagonal values to 0
    for idx in car_matrix.index:
        car_matrix.loc[idx, idx] = 0
    
    return car_matrix


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    # Add a new column 'car_type' based on 'car' values
    conditions = [
        (df['car'] <= 15),
        ((df['car'] > 15) & (df['car'] <= 25)),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(
        np.select(conditions, choices, default=''),
        index=df.index
    )
    
    # Calculate count of occurrences for each 'car_type'
    type_counts = df['car_type'].value_counts().to_dict()
    
    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))
    
    return sorted_type_counts


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    # Calculate the mean of the 'bus' column
    bus_mean = df['bus'].mean()
    
    # Find indexes where 'bus' values are greater than twice the mean
    indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    
    # Sort the indexes in ascending order
    sorted_indexes = sorted(indexes)
    
    return sorted_indexes


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    # Group DataFrame by 'route' and calculate average 'truck' values
    route_avg_truck = df.groupby('route')['truck'].mean()
    
    # Filter routes where average 'truck' values are greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()
    
    # Sort the list of routes in ascending order
    sorted_routes = sorted(filtered_routes)
    
    return sorted_routes


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    # Apply custom conditions to modify matrix values
    modified_matrix = matrix.copy()  # Create a copy of the matrix
    
    # Function to apply custom multiplication logic
    def custom_multiply(value):
        if value > 20:
            return round(value * 0.75, 1)
        else:
            return round(value * 1.25, 1)
    
    # Apply custom multiplication function to each element in the matrix
    modified_matrix = modified_matrix.map(custom_multiply)
    
    return modified_matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    # Combine 'startDay' and 'startTime' columns to create a start timestamp
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%A %H:%M:%S')
    
    # Combine 'endDay' and 'endTime' columns to create an end timestamp
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%A %H:%M:%S')
    
    # Calculate time differences
    df['time_difference'] = df['end_timestamp'] - df['start_timestamp']
    
    # Calculate the duration in hours
    df['duration_hours'] = df['time_difference'].dt.total_seconds() / 3600
    
    # Check if duration is within 24 hours and spans all 7 days for each ('id', 'id_2') pair
    check_conditions = (
        (df['duration_hours'] == 24 * 7) &  # 24 hours for 7 days
        (df['start_timestamp'].dt.time == pd.Timestamp('00:00:00').time()) &  # Start time is 00:00:00
        (df['end_timestamp'].dt.time == pd.Timestamp('23:59:59').time())  # End time is 23:59:59
    )
    
    # Group by 'id' and 'id_2' and check if any condition is False for each group
    time_completeness = ~check_conditions.groupby([df['id'], df['id_2']]).any()
    
    return time_completeness
