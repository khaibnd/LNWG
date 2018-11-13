#!usr/local/bin/python3.7

'''Build Gantt Chart by matplotlib'''

import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show


x = [1, 2, 3, 4, 5]
y = [6, 7, 8, 9, 10]

output_file('line.html')

p = figure(title='simple', x_axis_label='time', y_axis_label='to')
p.line(x, y, legend='Temp.', line_width=2)

show(p)