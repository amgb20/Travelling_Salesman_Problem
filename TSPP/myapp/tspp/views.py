import numpy as np
import itertools
import io
import base64

from django.shortcuts import render, redirect
from .forms import TSPForm
from . import SimpleGrid
from django.http import HttpResponse
from django.http import FileResponse

def home(request):
    return render(request, 'home.html')

def tspp_results(request):
    return render(request, 'tspp_results.html')

def path_coordinates_to_csv_string(path, coordinates):
    ordered_coordinates = coordinates[path]
    csv_data = "X,Y\n" + "\n".join([f"{coord[0]},{coord[1]}" for coord in ordered_coordinates])
    return csv_data

def download_path_csv(request, algorithm, Length, Width):
    Length = int(Length)
    Width = int(Width)
    path, _, _, _,_,_ = SimpleGrid.run(Length, Width, algorithm)
    coordinates = np.array(list(itertools.product(np.linspace(0, Length - 1, Length), np.linspace(0, Width - 1, Width))))
    csv_data = path_coordinates_to_csv_string(path, coordinates)
    
    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{algorithm}_{Length}x{Width}.csv"'
    
    return response

def download_elapsed_time_csv(request,algorithm, Length, Width):
    file_path = 'results.csv'
    response = FileResponse(open(file_path, 'rb'), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{algorithm}_{Length}x{Width}_Time_Complexity.csv"'
    return response

# --- RESTRUCTURE THIS FILE TO HAVE A BETTER FILE STRUCT --- #
def download_cpu_usages_csv(request,algorithm, Length, Width):
    file_path = 'results.csv'
    response = FileResponse(open(file_path, 'rb'), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{algorithm}_{Length}x{Width}_CPU_Usage.csv"'
    return response

# # Compile all three complexity plots into a single png file
# def download_all_complexitty_plots(request):
    
#     SimpleGrid.plot_all_complexity(10)

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
            path, cost, elapsed_time, image_base64, cpu_usages, memory_usage = SimpleGrid.run(Length, Width, tspp_algorithm=algorithm)
            csv_filename = f"{algorithm}_path.csv"
            
            plt_complexity = SimpleGrid.run_experiments_and_save_plot(Length,tspp_algorithm=algorithm)

             # Convert the plt_complexity instance to a base64 image
            buffer = io.BytesIO()
            plt_complexity.savefig(buffer, format='png')
            buffer.seek(0)
            image_complexity = base64.b64encode(buffer.getvalue()).decode('utf-8')

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

