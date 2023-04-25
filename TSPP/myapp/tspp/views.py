from django.shortcuts import render, redirect
from .forms import TSPForm
from . import SimpleGrid
import numpy as np
import itertools
import io
import base64

from django.http import HttpResponse

def path_coordinates_to_csv_string(path, coordinates):
    ordered_coordinates = coordinates[path]
    csv_data = "X,Y\n" + "\n".join([f"{coord[0]},{coord[1]}" for coord in ordered_coordinates])
    return csv_data

def download_csv(request, algorithm, Length, Width):
    Length = int(Length)
    Width = int(Width)
    path, _, _, _ = SimpleGrid.run(Length, Width, algorithm)
    coordinates = np.array(list(itertools.product(np.linspace(0, Length - 1, Length), np.linspace(0, Width - 1, Width))))
    csv_data = path_coordinates_to_csv_string(path, coordinates)
    
    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{algorithm}_{Length}x{Width}.csv"'
    
    return response


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
            path, cost, elapsed_time, image_base64 = SimpleGrid.run(Length, Width, tspp_algorithm=algorithm)
            csv_filename = f"{algorithm}_path.csv"
            
            plt_complexity = SimpleGrid.run_experiments_and_save_plot()

             # Convert the plt instance to a base64 image
            buffer = io.BytesIO()
            plt_complexity.savefig(buffer, format='png')
            buffer.seek(0)
            image_complexity = base64.b64encode(buffer.getvalue()).decode('utf-8')
            context['complexity_plot_path'] = image_complexity

            context['result'] = (path, cost, elapsed_time)
            context['image_base64'] = image_base64
            context['algorithm'] = algorithm
            context['Length'] = Length
            context['Width'] = Width
            context['csv_filename'] = f"{algorithm}_{Length}x{Width}.csv" 
        
    else:
        form = TSPForm()
    
    context['form'] = form
    return render(request, 'index.html', context)

# def result(request):
#     if request.method == 'POST':
#         form = TSPForm(request.POST)
#         if form.is_valid():
#             Lentgh = form.cleaned_data['Lentgh']
#             Width = form.cleaned_data['Width']
#             algorithm = form.cleaned_data['algorithm']

#             # Call SimpleGrid functions with the selected algorithm and grid_size
#             # Get the result and pass it to the template
#             result = SimpleGrid.run(Lentgh, Width, tspp_algorithm=algorithm)

#             return render(request, 'result.html', {'result': result})
#     return redirect('index')

