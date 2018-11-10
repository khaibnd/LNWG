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

        return parameter, demand, criteria, sequence, machine, machine_criteria, machine_lotsize, processing_time, wip, finished_goods


class DataOutput():

    def __init__(self, machine_criteria):
        self.machine_criteria = machine_criteria

    def data_writer(self,
                    output,
                    best_tardiness,
                    best_solution,
                    initial_finished_time,
                    genetic_finished_time):
        '''Write Output DataFrame to Excel file'''
        start = start_date()
        writer = pd.ExcelWriter(output)

        parameter = pd.DataFrame([['best_tardiness', best_tardiness],
                                  ['initial_finished_time', initial_finished_time],
                                  ['genetic_finished_time', genetic_finished_time]],
                                 columns=['criteria', 'value'])
        parameter.to_excel(writer, sheet_name='parameter')

        observation = FitnessCalculation.calculate_completion_time(self, best_solution.index.tolist(), best_solution)
        completion_time = observation['completion_time'].tolist

        start_time = {}

        def num_sunday(self, end, start, weekday=6):
            start = datetime.datetime.utcfromtimestamp(start)
            start_year = start.year
            start_month = start.month
            start_date = start.date

            end = datetime.datetime.utcfromtimestamp(end)
            end_year = end.year
            end_month = end.month
            end_date = end.date
            
            sunday, remainder = divmod((end - start).days, 7)
            if (weekday - start.weekday()) % 7 <= remainder:
                return sunday + 1
            else:
                return sunday
    
        for key in completion_time:
            
            completion_time[key] = completion_time[key] * 3600 + start + 24 * 60 * 60 * num_sunday(self, completion_time[key] * 3600 + start, start, 6)

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

    def operation_output_writer(self, filename):
        writer = pd.ExcelWriter(filename)
        for i in range(20):
            e = self.population_dict[i]
            e.to_excel(writer, sheet_name='s_%s' % i)
            writer.save()
        writer.save()

    def iteration_record_writer(self, iteration_output, iteration_record, best_solution):
        writer = pd.ExcelWriter(iteration_output)
        iteration_record.to_excel(writer, sheet_name='iteration')
        def num_sunday(self, end, start, weekday=6):
            start = datetime.datetime.utcfromtimestamp(start)
            start_date = start.date
            end = datetime.datetime.utcfromtimestamp(end)
            sunday, remainder = divmod((end - start).days, 7)

            if (weekday - start.weekday()) % 7 <= remainder:
                return sunday + 1
            else:
                return sunday
        
        
        def calculate_calendar_completion_time(self, row):
            start = start_date()
            calendar_ct = row['completion_time'] * 3600 + start +\
                            24 * 60 * 60 * num_sunday(self, row['completion_time'] * 3600 +start,
                                                      start, 6)
            calendar_ct = datetime.datetime.utcfromtimestamp(calendar_ct)
            return calendar_ct


        def calculate_calendar_start_time(self, row):
            start = start_date()
            start_time = row['completion_time'] - row['processing_time_plus']
            calendar_ct = start_time * 3600 + start +\
                            24 * 60 * 60 * num_sunday(self, start_time * 3600 +start,
                                                      start, 6)
            calendar_ct = datetime.datetime.utcfromtimestamp(calendar_ct)
            return calendar_ct

        best_solution_size = len(best_solution)
        best_solution['review'] = pd.Series([0] * best_solution_size)
        best_solution['min_assign'] = pd.Series([0] * best_solution_size)
        best_solution['max_assign'] = pd.Series([0] * best_solution_size)
        best_solution['completion_time'] = pd.Series([0.00] * best_solution_size)
        best_solution['machine_completion_time'] = pd.Series([0.00] * best_solution_size)
        best_solution['lot_completion_time'] = pd.Series([0.00] * best_solution_size)
        criteria_df = self.criteria.loc[:, ['part', 'coat_design', 'fabrication_part']]
        best_solution = best_solution.merge(criteria_df, on='part', how='left')
        
        best_solution['prev_operation'] = best_solution.apply(lambda row: 
                                                  FitnessCalculation.
                                                  prev_part_sequence_operation(self,
                                                                               row.part,
                                                                               row.operation,
                                                                               row.num_sequence), axis=1)
        best_solution['post_operation'] = best_solution.apply(lambda row:
                                                          FitnessCalculation.
                                                          post_part_sequence_operation(self,
                                                                                       row.part,
                                                                                       row.operation,
                                                                                       row.num_sequence), axis=1)

        best_solution['prev_operation_index'] = best_solution.apply(lambda row:
                                                                FitnessCalculation.
                                                                num_lot_operation_index(self,
                                                                                        row.num_lot,
                                                                                        row.prev_operation,
                                                                                        best_solution), axis=1)
        best_solution['post_operation_index'] = best_solution.apply(lambda row:
                                                                FitnessCalculation.
                                                                num_lot_operation_index(self,
                                                                                        row.num_lot,
                                                                                        row.post_operation,
                                                                                        best_solution), axis=1)
        
        best_solution['processing_time'] = best_solution.apply(lambda row:
                                                                FitnessCalculation.calculate_job_processing_time(self,
                                                                                                                 row.part,
                                                                                                                 row.operation,
                                                                                                                 row.num_sequence), axis=1)
        best_solution['processing_time_plus'] = best_solution.apply(lambda row:
                                                                FitnessCalculation.calculate_job_processing_time_plus(self,
                                                                                                                      row.processing_time,
                                                                                                                      row,
                                                                                                                      best_solution), axis=1)
        
        best_solution = FitnessCalculation.calculate_completion_time(self, best_solution.index.tolist(), best_solution)
        
        
        best_solution['calendar_completion_time'] = best_solution.apply(lambda row:
                                                                        calculate_calendar_completion_time(self,
                                                                                                           row), axis=1)
        best_solution['calendar_start_time'] = best_solution.apply(lambda row: 
                                                                   calculate_calendar_start_time(self,
                                                                                                 row), axis=1)

        best_solution = best_solution[['num_job', 'num_lot', 'part', 'operation', 'machine', 'num_sequence', 'max_assign', 'processing_time', 'processing_time_plus', 'calendar_start_time', 'calendar_completion_time']]

        best_solution.to_excel(writer, sheet_name='best_solution')
        writer.save()


class ExcelFile():

    def file_remove(self, file_link):
        from os import path, remove
        if path.exists(file_link):
            remove(file_link)
