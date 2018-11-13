'''Get data from excel file and convert to dataFrame'''
import datetime
import pandas as pd
from src.fitness_calculation.start_date import start_date
from src.fitness_calculation.fitness_calculation import FitnessCalculation


class DataInput():
    '''Get DataFrame Input from Excel file'''

    def data_reader(self, input_):
        '''Read excel file to dataFrame'''
        df_input = pd.ExcelFile(input_)
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
        machine_lotsize = sheet_to_df_map['machine_lotsize']
        finished_goods = sheet_to_df_map['finished_goods']

        return parameter, demand, criteria, sequence,\
               machine, machine_criteria, machine_lotsize,\
               processing_time, wip, finished_goods


class DataOutput():

    def __init__(self, machine_criteria):
        self.machine_criteria = machine_criteria


    def num_sunday(self, end, start, weekday=6):
        '''Get number of Sunday day between start and finish the job'''
        start = datetime.datetime.utcfromtimestamp(start)
        end = datetime.datetime.utcfromtimestamp(end)
        sunday, remainder = divmod((end - start).days, 7)

        if (weekday - start.weekday()) % 7 <= remainder:
            return sunday + 1
        else:
            return sunday


    def calculate_calendar_completion_time(self, row):
        '''Convert numberic completion time gene code to UTC'''
        start = start_date()
        time = row['completion_time']
        calendar_time = time * 3600 + start +\
                        24 * 60 * 60 * DataOutput.num_sunday(self, time * 3600 +start,
                                                             start, 6)
        calendar_time = datetime.datetime.utcfromtimestamp(calendar_time)
        return calendar_time


    def calculate_calendar_start_time(self, row):
        '''Convert numberic start time gene code to UTC'''
        start = start_date()
        time = row['completion_time'] - row['processing_time_plus']
        calendar_time = time * 3600 + start +\
                        24 * 60 * 60 * DataOutput.num_sunday(self, time * 3600 +start,
                                                             start, 6)
        calendar_time = datetime.datetime.utcfromtimestamp(calendar_time)
        return calendar_time


    def data_writer(self, best_tardiness,
                    best_solution, initial_finished_time,
                    genetic_finished_time, output_):
        '''Write Running Parameters to Excel file'''
        writer = pd.ExcelWriter(output_)

        parameter = pd.DataFrame([['best_tardiness', best_tardiness],
                                  ['initial_finished_time', initial_finished_time],
                                  ['genetic_finished_time', genetic_finished_time]],
                                 columns=['criteria', 'value'])
        parameter.to_excel(writer, sheet_name='parameter')

        '''Write the Global Best Solution to Excel file'''
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
                                                                                                   row,
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
                                       'num_sequence', 'max_assign', 'processing_time', 'processing_time_plus',
                                       'calendar_start_time', 'calendar_completion_time']]

        best_solution.to_excel(writer, sheet_name='best_solution')
        writer.save()

        return print('best_tardiness', best_tardiness)


    def iteration_record_writer(self, iteration_record, best_solution, iteration_output):
        '''Record itertion '''
        writer = pd.ExcelWriter(iteration_output)
        iteration_record.to_excel(writer, sheet_name='iteration')
        # Adding extention gene code
        best_solution = FitnessCalculation.adding_gene_code(self, best_solution)     
     
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
                                                                                                   row,
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
                                       'num_sequence', 'max_assign', 'processing_time', 'processing_time_plus',
                                       'calendar_start_time', 'calendar_completion_time']]

        best_solution.to_excel(writer, sheet_name='best_solution')
        writer.save()


    def operation_output_writer(self, population_dict, filename):
        '''Writer all population chromosome to Excel file'''
        writer = pd.ExcelWriter(filename)
        pop_len = len(population_dict)
        for i in range(pop_len):
            e = self.population_dict[i]
            e.to_excel(writer, sheet_name='s_%s' % i)
            writer.save()
        writer.save()


class ExcelFile():

    def file_remove(self, file_link):
        from os import path, remove
        if path.exists(file_link):
            remove(file_link)
