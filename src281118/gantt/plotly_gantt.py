'''Build Gantt Chart on http://plot.ly'''
import pandas as pd
import numpy as np
import plotly
import plotly.plotly as py
import plotly.figure_factory as ff
from src.fitness_calculation.start_date import start_date
from src.fitness_calculation.fitness_calculation import FitnessCalculation
from src.data.data import DataOutput

plotly.tools.set_credentials_file(username='khaibnd', api_key='pDObqrmhM1Mq6KqUKte1')
start_date = start_date()


class PlotlyGantt():
    '''Build production gantt-chart for machines in 
    https://plot.ly/~khaibnd/14/ln-wedge-lot-scheduling/#/'''

    def __init__(self, parameter):
        self.parameter = parameter

    def plotly_gantt(self, best_solution):
        '''Generating and uploading gantt chart data'''
        j_keys = best_solution['num_lot'].unique()
        # m_keys = best_solution['machine'].unique()
        gantt_dataframe = []

        # Adding extention gene code
        best_solution = FitnessCalculation.adding_gene_code(self, best_solution)     
        # Calculate each gene completion time     
        best_solution['processing_time'] = best_solution.apply(lambda row:
                                                                FitnessCalculation.
                                                                calculate_job_processing_time(self,
                                                                                              row.part,
                                                                                              row.operation,
                                                                                              row.num_sequence), axis=1)
        best_solution['processing_time_plus'] = best_solution.apply(lambda row:
                                                                FitnessCalculation.
                                                                calculate_job_processing_time_plus(self,
                                                                                                   row.processing_time,
                                                                                                   row.machine,
                                                                                                   row.num_lot,
                                                                                                   best_solution), axis=1)
        
        # Calculate each gene completion time
        best_solution = FitnessCalculation.calculate_completion_time(self, best_solution.index.tolist(), best_solution)

        # Convert start time & completion time to UTC time
        best_solution['calendar_completion_time'] = best_solution.apply(lambda row:
                                                                        DataOutput.
                                                                        calculate_calendar_completion_time(self,
                                                                                                           row), axis=1)
        best_solution['calendar_start_time'] = best_solution.apply(lambda row: 
                                                                   DataOutput.calculate_calendar_start_time(self,
                                                                                                            row), axis=1)
        #Slice the raw to the final report
        best_solution = best_solution[['num_job', 'num_lot', 'part', 'operation', 'machine',
                                       'num_sequence', 'lotsize_assign', 'processing_time', 'processing_time_plus',
                                       'calendar_start_time', 'calendar_completion_time']]

        for _, row in best_solution.iterrows():
            num_job = row['num_job']
            machine = row['machine']
            gantt_finish = row['calendar_completion_time']

            gantt_start = row['calendar_start_time']

            gantt_dataframe.append(dict(Task='Machine %s' % (machine),
                                        Start='%s' % (str(gantt_start)),
                                        Finish='%s' % (str(gantt_finish)),
                                        Resource='Job_%s' % (num_job)))

        color = {'Job_%s' % (k):'rgb' + str(tuple(np.random.choice(range(256), size=3)))
                 for k in range(len(j_keys))}
        fig = ff.create_gantt(gantt_dataframe,
                              index_col='Resource',
                              colors=color,
                              show_colorbar=True,
                              group_tasks=True,
                              showgrid_x=True,
                              title='LN Wedge Lot Scheduling')
        return py.plot(fig, filename='LN Wedge Lot Scheduling', world_readable=True)
