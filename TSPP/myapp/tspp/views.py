import numpy as np
import itertools
import io
import base64
import json
import math
import random
import csv

from django.shortcuts import render, redirect
from django.conf import settings
from .forms import TSPForm
from . import SimpleGrid
from django.http import HttpResponse
from django.http import FileResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from geopy.distance import geodesic

from math import radians, sin, cos, asin, sqrt, atan2

# addiding the cvrp.py file
from .ROMIE_Out_Of_Charge_Points import compute_out_of_charge_points
from shapely.geometry import Polygon


def home(request):
    return render(request, 'home.html')


def tspp_results(request):
    return render(request, 'tspp_results.html')


def mapview(request):
    context = {'GOOGLE_MAPS_API_KEY': settings.GOOGLE_MAPS_API_KEY}
    return render(request, 'map.html', context)


def path_coordinates_to_csv_string(path, coordinates):
    ordered_coordinates = coordinates[path]
    csv_data = "X,Y\n" + \
        "\n".join([f"{coord[0]},{coord[1]}" for coord in ordered_coordinates])
    return csv_data


def download_path_csv(request, algorithm, Length, Width):
    Length = int(Length)
    Width = int(Width)
    path, _, _, _, _, _, _, = SimpleGrid.run(
        Length, Width, algorithm, max_distance=10)  # added last argument
    coordinates = np.array(list(itertools.product(np.linspace(
        0, Length - 1, Length), np.linspace(0, Width - 1, Width))))
    csv_data = path_coordinates_to_csv_string(path, coordinates)

    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{algorithm}_{Length}x{Width}.csv"'

    return response


def download_elapsed_time_csv(request, algorithm, Length, Width):
    file_path = 'results.csv'
    response = FileResponse(open(file_path, 'rb'), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{algorithm}_{Length}x{Width}_Time_Complexity.csv"'
    return response

# --- RESTRUCTURE THIS FILE TO HAVE A BETTER FILE STRUCT --- #


def download_cpu_usages_csv(request, algorithm, Length, Width):
    file_path = 'results.csv'
    response = FileResponse(open(file_path, 'rb'), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{algorithm}_{Length}x{Width}_CPU_Usage.csv"'
    return response

# # Compile all three complexity plots into a single png file
# def download_all_complexitty_plots(request):

#     SimpleGrid.plot_all_complexity(10)

# script solving the TSP problem using OR-Tools


# https://developers.google.com/optimization/routing/tsp -- this comes from directly from OR-tools a google tsp solver developed in python
@csrf_exempt
def solve_tsp(request):
    if request.method == "POST":
        data = json.loads(request.body)
        locations = data['locations']
        capacity = data['capacity']
        # rectangle_bound = data['rectangle_bounds']

        # Create distance matrix
        distance_matrix = []
        for location_1 in locations:  # for each location in the list of locations
            row = []
            for location_2 in locations:  # for each location in the list of locations
                row.append(int(geodesic(
                    (location_1['lat'], location_1['lng']), (location_2['lat'], location_2['lng'])).meters))  # append the distance between the two locations
            distance_matrix.append(row)

            print('distance_matrix', distance_matrix)


        # Save distance matrix to a CSV file
        filename = 'distance_matrix.csv'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(distance_matrix)

        print(f'Distance matrix saved to {filename}')

        # Create data model
        data = {}
        data['distance_matrix'] = distance_matrix
        data['num_vehicles'] = 1
        data['depot'] = 0

        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']), data['num_vehicles'], data['depot'])

        # Create routing model
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_callback_index = routing.RegisterTransitCallback(
            distance_callback)

        # Define cost of each arc
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # ------------------ For First solution strategy ------------------ #
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            route = []
            distances = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(locations[manager.IndexToNode(index)])
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(index):
                    distances.append(data['distance_matrix']
                                     [previous_index][index])

            # added line
            out_of_charge_points = compute_out_of_charge_points(
                route, distances, capacity)

            # Calculate the centroid of the out of charge points
            # https://en.wikipedia.org/wiki/Centroid
            # def calculate_centroid(points):
            #     lats = [p['lat'] for p in points]
            #     lngs = [p['lng'] for p in points]
            #     centroid = {'lat': sum(lats) / len(points), 'lng': sum(lngs) / len(points)}
            #     return centroid
            # https://stackoverflow.com/questions/2792443/finding-the-centroid-of-a-polygon

            def calculate_polygon_centroid(points):
                # if points is inferior or equal to 4, then the centroid is the average of the points
                if len(points) <= 4:
                    lats = [p['lat'] for p in points]
                    lngs = [p['lng'] for p in points]
                    centroid = {'lat': sum(
                        lats) / len(points), 'lng': sum(lngs) / len(points)}
                    return centroid
                else:
                    area = 0.0
                    x_center = 0.0
                    y_center = 0.0

                    a = len(points)
                    # repeat the first point to create a 'closed loop'
                    points = points + [points[0]]

                    for i in range(a):
                        xi, yi = points[i]['lat'], points[i]['lng']
                        xi1, yi1 = points[i + 1]['lat'], points[i + 1]['lng']

                        # calculate the signed area contribution of the current point
                        fi = xi * yi1 - xi1 * yi
                        area += fi
                        x_center += (xi + xi1) * fi
                        y_center += (yi + yi1) * fi

                    area *= 0.5

                    if area == 0.0:
                        return None

                    x_center /= (6.0 * area)
                    y_center /= (6.0 * area)

                    return {'lat': x_center, 'lng': y_center}

            charging_station_location = calculate_polygon_centroid(
                out_of_charge_points)

            # Calculate the total cost by summing the distances
            total_cost = sum(distances)

            def calculate_distance(point1, point2):
                lat1 = radians(point1['lat'])
                lon1 = radians(point1['lng'])
                lat2 = radians(point2['lat'])
                lon2 = radians(point2['lng'])

                # Radius of the Earth in kilometers
                earth_radius = 6371.0

                # Haversine formula
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                distance = earth_radius * c

                return distance*2

            # Calculate the total distance from each out_of_charge point to the charging station
            total_distance = 0.0
            for point in out_of_charge_points:
                distance_to_charging_station = calculate_distance(
                    point, charging_station_location)
                total_distance += distance_to_charging_station

            print('out_of_charge_points', out_of_charge_points)
            print('charging_station_location', charging_station_location)
            print('distances', distances)
            print('cost', total_cost)
            print('ooc-cs', total_distance)

            return JsonResponse({
                'route': route,
                'distances': distances,
                'cost': total_cost,
                'out_of_charge_points': out_of_charge_points,
                'charging_station_location': charging_station_location,
            })

    return JsonResponse({'error': 'Invalid request'}, status=400)

class IterationMonitor(pywrapcp.SearchMonitor):
    def __init__(self, solver, routing, manager, data):
        super().__init__(solver)
        self.routing = routing
        self.manager = manager
        self.data = data
        self.iterations = []
        self.costs = []

    def BeginNextDecision(self, b):
        # Record the iteration number and the cost of the current solution
        index = self.routing.Start(0)
        cost = 0
        while not self.routing.IsEnd(index):
            cost += self.data['distance_matrix'][self.manager.IndexToNode(index)][self.manager.IndexToNode(self.routing.NextVar(index).Value())]
            index = self.routing.NextVar(index).Value()
        self.iterations.append(self.solver().iterations())
        self.costs.append(cost)
        return True  # Continue the search


def index(request):
    result = None
    form = TSPForm()
    context = {}

    if request.method == 'POST':
        form = TSPForm(request.POST)

        if form.is_valid():
            Length = form.cleaned_data['Length']
            Width = form.cleaned_data['Width']
            algorithm = form.cleaned_data['algorithm']

            # Call SimpleGrid functions with the selected algorithm and grid_size
            # Get the result and pass it to the template
            path, cost, elapsed_time, image_base64, cpu_usages, memory_usage, chargin_station = SimpleGrid.run(
                Length, Width, tspp_algorithm=algorithm, max_distance=10)
            csv_filename = f"{algorithm}_path.csv"

            plt_complexity = SimpleGrid.run_experiments_and_save_plot(
                Length, tspp_algorithm=algorithm)

            # Convert the plt_complexity instance to a base64 image
            buffer = io.BytesIO()
            plt_complexity.savefig(buffer, format='png')
            buffer.seek(0)
            image_complexity = base64.b64encode(
                buffer.getvalue()).decode('utf-8')

            # plt_para = SimpleGrid.run_parallel_experiments(Length,tspp_algorithm=algorithm)
            # # Convert the plt_para instance to a base64 image
            # buffer1 = io.BytesIO()
            # plt_para.savefig(buffer1, format='png')
            # buffer1.seek(0)
            # image_para = base64.b64encode(buffer1.getvalue()).decode('utf-8')

            # create all the context objects
            context['complexity_plot_path'] = image_complexity
            # context['para_plot_path'] = image_para
            context['result'] = (path, cost, elapsed_time)
            context['image_base64'] = image_base64
            context['algorithm'] = algorithm
            context['Length'] = Length
            context['Width'] = Width
            context['csv_filename'] = f"{algorithm}_{Length}x{Width}.csv"
            context['png_filename'] = f"{algorithm}_{Length}x{Width}.png"

    else:
        form = TSPForm()

    context['form'] = form
    return render(request, 'index.html', context)
