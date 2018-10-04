'''Get data from excel file and convert to dataFrame'''
import datetime
import pandas as pd
from src.fitness_calculation.start_date import start_date
from src.fitness_calculation.fitness_calculation import FitnessCalculation


class DataInput():
    '''Get DataFrame Input from Excel file'''
    def data_reader(self, input):
        '''Read excel file to dataFrame'''
        df_input = pd.ExcelFile(input)
        sheet_to_df_map = {}
        for sheet_name in df_input.sheet_names:
            sheet_to_df_map[sheet_name] = df_input.parse(sheet_name)

        parameter = sheet_to_df_map['parameter']
        demand = sheet_to_df_map['demand']
        criteria = sheet_to_df_map['criteria']
        sequence = sheet_to_df_map['sequence']
        processing_time = sheet_to_df_map['processing_time']
        wip = sheet_to_df_map['wip']
        machine = sheet_to_df_map['machine']
        machine_criteria = sheet_to_df_map['machine_criteria']
        finished_goods = sheet_to_df_map['finished_goods']

        return parameter, demand, criteria, sequence, machine, machine_criteria, processing_time, wip, finished_goods


class DataOutput():

    def data_writer(self, output, best_makespan, best_solution):
        '''Write Output DataFrame to Excel file'''
        start = start_date()
        writer = pd.ExcelWriter(output)

        parameter = pd.DataFrame([['best_makespan', best_makespan],
                                  ['best_time', 2]],
                                 columns=['criteria', 'value'])
        parameter.to_excel(writer, sheet_name='parameter')

        _, __, completion_time = FitnessCalculation.calculate_finished_time(self, best_solution)

        start_time = {}
        for key in completion_time:
            completion_time[key] = completion_time[key] * 3600 + start

            part = best_solution.loc[best_solution.index == key]['part'].values[0]
            operation = best_solution.loc[best_solution.index == key]['operation'].values[0]
            num_sequence = best_solution.loc[best_solution.index == key]['num_sequence'].values[0]
            job_run_time = FitnessCalculation.calculate_job_processing_time(self, part, operation, num_sequence)
            start_time[key] = completion_time[key] - job_run_time * 60 * 60
            completion_time[key] = datetime.datetime.utcfromtimestamp(completion_time[key])
            start_time[key] = datetime.datetime.utcfromtimestamp(start_time[key])

        completion_time = pd.Series(completion_time)
        best_solution = pd.concat([best_solution,
                                   completion_time.rename('completion_time')], axis=1)
        
        start_time = pd.Series(start_time) 
        best_solution = pd.concat([best_solution, start_time.rename('start_time')], axis=1)
        best_solution = best_solution[['num_job', 'num_lot', 'part', 'operation', 'machine', 'num_sequence', 'start_time', 'completion_time']]
        best_solution.to_excel(writer, sheet_name='best_solution')
        writer.save()

        return best_solution
