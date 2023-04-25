from django.shortcuts import render, redirect
from .forms import TSPForm
from . import SimpleGrid

# def index(request):
#     form = TSPForm()
#     return render(request, 'index.html', {'form': form})

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

            context['result'] = (path, cost, elapsed_time)
            context['image_base64'] = image_base64
            context['csv_filename'] = f"{algorithm}_{Length}x{Width}.csv" 
            # context = {'form':form, 'result': result, 'image_base64': image_base64, 'csv_filename': csv_filename}
        
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

