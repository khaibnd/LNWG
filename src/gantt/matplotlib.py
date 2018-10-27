'''Build Gantt Chart on http://plot.ly'''
import pandas as pd
import numpy as np
from src.gantt import plotly_gantt
import plotly.plotly as py
import plotly.figure_factory as ff

import matplotlib.pyplot as plt

from src.fitness_calculation.start_date import start_date
from src.fitness_calculation.fitness_calculation import fitness_calculation

plotly_gantt.tools.set_credentials_file(username='khaibnd', api_key='pDObqrmhM1Mq6KqUKte1')
start_date = start_date()


class MatplotlibGantt():
    '''Initial input'''

    def __init__(self, parameter):
        self.parameter = parameter

    def plotly_gantt(self, best_solution):
        '''Generating and uploading gantt chart data'''
        j_keys = best_solution['num_lot'].unique()
        # m_keys = best_solution['machine'].unique()
        gantt_dataframe = []

        print(best_solution)
        _, __, completion_time = fitness_calculation.calculate_finished_time(self, best_solution)
        print(completion_time)

        for row_index, row in best_solution.iterrows():
            part = row['part']
            num_job = row['num_job']
            operation = row['operation']
            machine = row['machine']

            job_run_time = fitness_calculation.calculate_job_processing_time(self,
                                                                             part,
                                                                             num_job,
                                                                             operation,
                                                                             machine)
            finish = completion_time[row_index] * 60 * 60 + start_date
            gantt_finish = pd.to_datetime(finish, unit='s')
            start = finish - job_run_time * 60 * 60
            gantt_start = pd.to_datetime(start, unit='s')

            gantt_dataframe.append(dict(Task='Machine %s' % (machine),
                                        Start='%s' % (str(gantt_start)),
                                        Finish='%s' % (str(gantt_finish)),
                                        Resource='Job_%s' % (num_job)))

        color = {'Job_%s' % (k):'rgb' + str(tuple(np.random.choice(range(256), size=3)))
                 for k in range(len(j_keys))}
        fig = plt.figure()
        
        '''
        fig = ff.create_gantt(gantt_dataframe,
                              index_col='Resource',
                              colors=color,
                              show_colorbar=True,
                              group_tasks=True,
                              showgrid_x=True,
                              title='LN Wedge Lot Scheduling')
        return py.plot(fig, filename='LN Wedge Lot Scheduling', world_readable=True)
        '''
