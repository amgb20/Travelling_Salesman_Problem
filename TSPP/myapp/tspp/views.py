from django.shortcuts import render, redirect
from .forms import TSPForm
from . import SimpleGrid

# def index(request):
#     form = TSPForm()
#     return render(request, 'index.html', {'form': form})

def index(request):
    result = None
    form = TSPForm()

    if request.method == 'POST':
        form = TSPForm(request.POST)

        if form.is_valid():
            Length = form.cleaned_data['Length']
            Width = form.cleaned_data['Width']
            algorithm = form.cleaned_data['algorithm']

            # Call SimpleGrid functions with the selected algorithm and grid_size
            # Get the result and pass it to the template
            result = SimpleGrid.run(Length, Width, tspp_algorithm=algorithm)
    else:
        form = TSPForm()

    return render(request, 'index.html', {'form':form, 'result': result})

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

