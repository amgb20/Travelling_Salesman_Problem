from geopy.distance import geodesic
import numpy as np

"""
    Compute out_of_charge points and charging station location.

    Parameters:
    - route: A list of waypoints from the TSP solver.
    - capacity: ROMIE's battery capacity in meters.
    - rectangle_bounds: A dictionary with 'lat_min', 'lat_max', 'lng_min', 'lng_max' as keys.

    Returns:
    - charging_station: A dictionary with 'lat' and 'lng' as keys for the charging station location.
    - out_of_charge_points: A list of dictionaries with 'lat' and 'lng' as keys for out_of_charge points.
    """

def compute_charge_points(route, distances, capacity, rectangle_bounds):
    out_of_charge_points = []
    remaining_capacity = capacity
    last_station_location = route[0]  # Starts fully charged at the depot
    current_location = last_station_location

    for i in range(len(distances)):
        distance = distances[i]
        while remaining_capacity < distance:
            # Compute the ratio of the distance that can be covered before running out of charge
            ratio = remaining_capacity / distance
            # Compute the out-of-charge point
            out_of_charge_point = {
                'lat': current_location['lat'] + ratio * (route[i+1]['lat'] - current_location['lat']),
                'lng': current_location['lng'] + ratio * (route[i+1]['lng'] - current_location['lng']),
            }
            out_of_charge_points.append(out_of_charge_point)
            remaining_capacity = capacity  # The robot is recharged
            # The robot will continue its path from the out_of_charge_point
            current_location = out_of_charge_point
            # Subtract the already covered distance from the total distance
            distance = distance - (ratio * distance)
        # Subtract the distance to the next waypoint from the remaining capacity
        remaining_capacity -= distance
        # Move to the next waypoint
        current_location = route[i+1]

    # Use an optimization approach to find the best location for the charging station within the rectangle bounds.
    # This is a placeholder for the actual optimization code.
    charging_station = {
        'lat': (rectangle_bounds['lat_min'] + rectangle_bounds['lat_max']) / 2,
        'lng': (rectangle_bounds['lng_min'] + rectangle_bounds['lng_max']) / 2,
    }

    return charging_station, out_of_charge_points



