import numpy as np
import itertools
import io
import base64
import json
import math

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

        # Create distance matrix
        distance_matrix = []
        for location_1 in locations:
            row = []
            for location_2 in locations:
                row.append(int(geodesic(
                    (location_1['lat'], location_1['lng']), (location_2['lat'], location_2['lng'])).meters))
            distance_matrix.append(row)

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

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(locations[manager.IndexToNode(index)])
                previous_index = index
                index = solution.Value(routing.NextVar(index))

            return JsonResponse({
                'route': route,
                'cost': solution.ObjectiveValue(),
            })

    return JsonResponse({'error': 'Invalid request'}, status=400)


def compute_distance_matrix(locations):
    num_locations = len(locations)
    dist_matrix = []
    ordered_points = []

    for location in locations:
        ordered_points.append(location)

    for i in range(num_locations):
        dist_matrix_row = []
        for j in range(num_locations):
            if i == j:
                dist_matrix_row.append(0)
            else:
                lat1, lon1 = ordered_points[i]['lat'], ordered_points[i]['lng']
                lat2, lon2 = ordered_points[j]['lat'], ordered_points[j]['lng']
                radius = 6371  # radius of the Earth in kilometers

                dlat = math.radians(lat2-lat1)
                dlon = math.radians(lon2-lon1)

                a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
                    * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

                distance = radius * c * 1000  # Convert the distance to meters
                dist_matrix_row.append(distance)
        dist_matrix.append(dist_matrix_row)

    return dist_matrix, ordered_points


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
