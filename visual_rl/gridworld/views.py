from django.shortcuts import render
from django.http import HttpRequest, HttpResponse


def tabular_grid(request: HttpRequest) -> HttpResponse:
    return render(request, 'tabular_grid.html')

def add(request: HttpRequest) -> HttpResponse:
    # grid_row = request.GET.get('grid_row')
    grid_row = request.GET['num1']
    grid_col = request.GET['num2']

    result = int(grid_row) + int(grid_col)


    return render(request, 'add.html', {'grid_row': grid_row, 'grid_col': grid_col, 'result': result})

