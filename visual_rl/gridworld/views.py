import re
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse

from algorl.src.grid_environment import Make

def tabular_grid(request: HttpRequest) -> HttpResponse:
    return render(request, 'tabular_grid.html')

def add(request: HttpRequest) -> HttpResponse:
    grid_row = request.GET['grid_row'] # request.GET.get('grid_row')
    grid_col = request.GET['grid_col']

    walls = request.GET['walls']
    breakdown = re.findall('\(.*?,.*?\)', walls)
    walls = [tuple(map(int, list(i[1:-1].split(',')))) for i in breakdown]

    env = Make(grid_col=int(grid_col), grid_row=int(grid_row), walls=walls)

    return render(request, 'add.html', {'grid_row': grid_row, 'grid_col': grid_col, 'result': env.grid})

