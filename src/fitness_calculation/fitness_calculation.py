'''Calculate production leadtime per observation'''
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from src.initial_solution.initial_solution import InitialSolution


class FitnessCalculation():
    '''Measure invidial demand job completion time and Total Weighed Tardiness'''

    def __init__(self, sequence, machine_lotsize):
        self.sequence = sequence
        self.machine_lotsize = machine_lotsize

    def prev_part_sequence_operation(self, part, operation, num_sequence):
        part_sequence = InitialSolution.part_sequence(self, part, num_sequence)
        operation_index = part_sequence.index(operation)
        if (operation_index - 1 >= 0):
            try:
                prev_part_sequence_operation = part_sequence[operation_index - 1]
            except:
                prev_part_sequence_operation = None
        else:
            prev_part_sequence_operation = None
        return prev_part_sequence_operation

    def post_part_sequence_operation(self, part, operation, num_sequence):
        part_sequence = InitialSolution.part_sequence(self, part, num_sequence)
        operation_index = part_sequence.index(operation)
        try:
            post_part_sequence_operation = part_sequence[operation_index + 1]
        except:
            post_part_sequence_operation = None
        return post_part_sequence_operation
    
    
    def num_lot_operation_index(self, num_lot, operation, observation):
        try:
            index = observation.index[np.logical_and(observation.num_lot == num_lot,
                                                          observation.operation == operation)].tolist()
            index = index[0]
        except:                                 
            index = None
        return index
        
    
    def calculate_job_processing_time(self, part, operation, num_sequence):
        '''Completion time each num_lot(= batch processing time)'''
        try:
            def line_index(num_sequence, part):
                line_index = self.processing_time.index[(self.processing_time['num_sequence'] == num_sequence)
                                                  & (self.processing_time['part'] == part)].tolist()
                # Get the smallest index
                line_index = line_index[0]
                return line_index
    
            job_processing_time = self.processing_time.at[line_index(num_sequence, part), operation]
        except:
            print('Wrong processing time!')
            print('part: %s, operation: %s, num_sequence: %s' % (part, operation, num_sequence))
            
        return job_processing_time

 
    def gene_ga_calculation(self, job_processing_time, row, observation):

        machine = row['machine']
        num_lot = row['num_lot']
        machine_daily_work_hour = self.machine_criteria.loc[self.machine_criteria.machine == machine, 'work_hour'].values[0]
        
        #print('origin:', row)
        machine_completion_time = observation.loc[observation.machine == machine][ 'machine_completion_time'].max()

        lot_completion_time = observation.loc[observation.num_lot == num_lot]['lot_completion_time'].max()

        
        completion_time = max(machine_completion_time, lot_completion_time) + job_processing_time
        
        # Adding extra hours
        try:
            num_day = math.floor((completion_time -  machine_completion_time)/ machine_daily_work_hour)
        except:
            num_day = 0
        free_hour = completion_time -  machine_completion_time - job_processing_time

        if num_day > 0 and free_hour == 0:
            completion_time = completion_time + num_day * (24 - machine_daily_work_hour)
        
        elif num_day > 0 and free_hour <= num_day * (24 - machine_daily_work_hour):
            completion_time = completion_time + num_day * (24 - machine_daily_work_hour) - free_hour

        return completion_time

    def line_index(self, machine, num_sequence):
        try:
            line_index = self.machine_lotsize.index[(self.machine_lotsize.machine == machine)
                                                      & (self.machine_lotsize.num_sequence == num_sequence)].tolist()
            line_index = line_index[0]
            
        except(IndexError):
            print('IndexError: list index out of range')
            print('Combination %s and %s is not in the input file' % (machine, num_sequence))
        return line_index


    def get_groupable_df(self, part, coat_design, fabrication_part, operation, machine, check_df):

        check_df = check_df.loc[np.logical_and(check_df.completion_time == 0, check_df.machine == machine)]

        if (operation == 'coat'):
            coat_design = self.criteria.loc[self.criteria.part == part, 'coat_design'].values[0]

            groupable_df = check_df[(check_df.operation == 'coat') &
                                    (check_df.coat_design == coat_design)]
            return groupable_df
        
        elif (operation == 'block_wedge'):
            coat_design = self.criteria.loc[self.criteria.part == part, 'coat_design'].values[0]

            groupable_df = check_df[(check_df.operation == 'block_wedge') &
                                    (check_df.coat_design == coat_design)]
            return groupable_df
    
        elif (operation == 'rob_slicing'):
            fabrication_part = self.criteria.loc[self.criteria.part == part, 'fabrication_part'].values[0]
    
            groupable_df = check_df[np.logical_and(check_df.operation == 'rob_slicing',
                                                   check_df.fabrication_part == fabrication_part)]
            return groupable_df
    
        elif (operation != 'rob_slicing' and
              operation != 'coat' and
              operation != 'block_wedge'):
    
            groupable_df = check_df[(check_df.part == part) &
                                    (check_df.operation == operation)]
            return groupable_df
        else:
            print('check group df condition')


    def get_different_machine_groupable_df(self, part, coat_design, fabrication_part, operation, check_df):

        criteria_df = self.criteria.loc[:, ['part', 'coat_design', 'fabrication_part']]
        check_df = check_df.merge(criteria_df, on='part', how='left')
        check_df = check_df.loc[check_df.completion_time == 0]

        if (operation == 'coat'):
            coat_design = self.criteria.loc[self.criteria.part == part, 'coat_design'].values[0]

            groupable_df = check_df[(check_df.operation == 'coat') &
                                    (check_df.coat_design == coat_design)]
            return groupable_df

        elif (operation == 'block_wedge'):
            coat_design = self.criteria.loc[self.criteria.part == part, 'coat_design'].values[0]

            groupable_df = check_df[(check_df.operation == 'block_wedge') &
                                    (check_df.coat_design == coat_design)]
            return groupable_df

        elif (operation == 'rob_slicing'):
            fabrication_part = self.criteria.loc[self.criteria.part == part, 'fabrication_part'].values[0]
    
            groupable_df = check_df[(check_df.operation == 'rob_slicing') &
                                    (check_df.fabrication_part == fabrication_part)]
            return groupable_df
    
        elif (operation != 'rob_slicing' and
              operation != 'coat' and
              operation != 'block_wedge'):
    
            groupable_df = check_df[(check_df.part == part) &
                                    (check_df.operation == operation)]
            return groupable_df
        else:
            print('check group df condition')


    def assign_condition(self, row, observation):
        if pd.isnull(row['prev_operation_index']):
            return True
        elif observation.at[int(row['prev_operation_index']), 'completion_time'] > 0:
            return True
        else:
            False


    def calculate_finished_time(self, observation):
        '''CALCULATE'''
        
        observation_size = len(observation)
        observation['min_assign'] = pd.Series([0] * observation_size)
        observation['max_assign'] = pd.Series([0] * observation_size)
        observation['completion_time'] = pd.Series([0.00] * observation_size)
        observation['machine_completion_time'] = pd.Series([0.00] * observation_size)
        observation['lot_completion_time'] = pd.Series([0.00] * observation_size)

        criteria_df = self.criteria.loc[:, ['part', 'coat_design', 'fabrication_part']]
        observation = observation.merge(criteria_df, on='part', how='left')
        observation['prev_operation'] = observation.apply(lambda row: 
                                                          FitnessCalculation.
                                                          prev_part_sequence_operation(self,
                                                                                       row.part,
                                                                                       row.operation,
                                                                                       row.num_sequence), axis=1)
        observation['post_operation'] = observation.apply(lambda row:
                                                          FitnessCalculation.
                                                          post_part_sequence_operation(self,
                                                                                       row.part,
                                                                                       row.operation,
                                                                                       row.num_sequence), axis=1)

        observation['prev_operation_index'] = observation.apply(lambda row:
                                                                FitnessCalculation.
                                                                num_lot_operation_index(self,
                                                                                        row.num_lot,
                                                                                        row.prev_operation,
                                                                                        observation), axis=1)
        observation['post_operation_index'] = observation.apply(lambda row:
                                                                FitnessCalculation.
                                                                num_lot_operation_index(self,
                                                                                        row.num_lot,
                                                                                        row.post_operation,
                                                                                        observation), axis=1)
        
        writer = pd.ExcelWriter(r'/Users/khaibnd/github-repositories/LNWG/src/data/new_output.xlsx')
        observation.to_excel(writer, sheet_name='s')
        writer.save()
        for row_index, row in observation.iterrows():
            '''Start main calculation'''
            #print('row_index', row_index)
            # Check the job yet observe
            if (np.logical_and(observation.at[row_index, 'completion_time'] == 0,
                               FitnessCalculation.assign_condition(self, row, observation))):
                part = row['part']
                operation = row['operation']
                machine = row['machine']
                num_sequence = row['num_sequence']
                coat_design = row['coat_design']
                fabrication_part = row['fabrication_part']
                #print(coat_design)
                max_lotsize = int(self.machine_lotsize.at[FitnessCalculation.line_index(self,
                                                                                        machine,
                                                                                        num_sequence),
                                                                                        'max_lotsize'])
                min_lotsize = int(self.machine_lotsize.at[FitnessCalculation.line_index(self,
                                                                                        machine,
                                                                                        num_sequence),
                                                                                        'min_lotsize'])
                job_processing_time = FitnessCalculation.calculate_job_processing_time(self,
                                                                                       part,
                                                                                       operation,
                                                                                       num_sequence)
                completion_time = FitnessCalculation.gene_ga_calculation(self,
                                                                         job_processing_time,
                                                                         row, observation)
                # Step 6.1: Machine load lotsize = 1
                if (np.logical_and(max_lotsize == 1, min_lotsize == 1)):
                    observation.at[row_index, ['completion_time',\
                                               'machine_completion_time',
                                               'lot_completion_time']] = completion_time
                    observation.at[row_index, ['min_assign', 'max_assign']] = 1

                # Step 6.2: maximum lotsize > 1 and minimum lotsize = 1
                elif (np.logical_and(max_lotsize > 1, min_lotsize == 1)):
                    succeed_observation = observation.loc[row_index:, :]
                    groupable_df = FitnessCalculation.get_groupable_df(self,
                                                                       part, coat_design, fabrication_part, operation,
                                                                       machine, succeed_observation)

                    #The group index is first operation of sequence
                    if (pd.isnull(row['prev_operation_index'])):
                        #print('max first operation of the lot')
                        assign_df = groupable_df

                        #Cách tăng tốc độ assignment bằng np.where
                        #observation['completion_time'] = np.where(observation.index.isin([assign_df_index]), completion_time, observation['completion_time'])
                    else:
                        #print('max grouping at middle')
                        assign_df = groupable_df.loc[np.logical_or(groupable_df.prev_operation_index < row_index,
                                                                   pd.isnull(groupable_df.prev_operation_index))]
                    assign_df_index = assign_df.index[:max_lotsize].tolist()
                    lot_size = len(assign_df_index)
                    observation.at[assign_df_index, 'max_assign'] = lot_size
                    observation.at[assign_df_index, 'min_assign'] = 1
                    observation.at[assign_df_index, ['completion_time',\
                                                     'machine_completion_time',
                                                     'lot_completion_time']] = completion_time
                
                # Step 6.3: minimum lotsize = minimum lotsize > 1
                elif (min_lotsize == max_lotsize):

                    precede_observation = observation.loc[np.logical_and(observation.completion_time == 0,
                                                                         observation.index <= row_index)]
                    assign_df = FitnessCalculation.get_groupable_df(self, part, coat_design, fabrication_part, operation, machine, precede_observation)
                    assign_df_size = len(assign_df)
                    
                    

                    # Previous contain enough lot for minimum lotsize load
                    if assign_df_size == min_lotsize:
                        #print('min full assign same machine')
                        assign_df_index = assign_df.index.tolist()
                        observation.loc[assign_df_index, ['min_assign', 'max_assign']] = assign_df_size
                        #print(min_lotsize)
                        #print(observation.loc[min_lotsize_index,['min_assign', 'max_assign']])
                        # Con thieu dieu kien can max_lotsize phai la max hien tai va cai moi
                        observation.at[assign_df_index, ['completion_time',\
                                                         'machine_completion_time',
                                                         'lot_completion_time']] = completion_time
                        
                        recheck_lot = assign_df['num_lot'].unique()
                        
                        recheck_df = observation.loc[np.logical_and(observation.num_lot.isin(recheck_lot),
                                                                    observation.completion_time == 0)]
                        recheck_df = FitnessCalculation.calculate_finished_time(self, recheck_df)
                        observation.update(recheck_df)
                        # Get list of lot can re-calculate completion time
                    else:
                        assign_df = FitnessCalculation.get_different_machine_groupable_df(self, part, coat_design, fabrication_part, operation, precede_observation)
                        assign_df_size = len(assign_df)
                        
                        if(assign_df_size == 2 * min_lotsize):
                            assign_df = assign_df[:min_lotsize]
                            #print('min full assign different machine')
                            assign_df_index = assign_df.index.tolist()
                            observation.loc[assign_df_index, ['min_assign', 'max_assign']] = assign_df_size
                            #print(min_lotsize)
                            #print(observation.loc[min_lotsize_index,['min_assign', 'max_assign']])
                            # Con thieu dieu kien can max_lotsize phai la max hien tai va cai moi
                            observation.at[assign_df_index, ['completion_time',\
                                                             'machine_completion_time',
                                                             'lot_completion_time']] = completion_time
                            
                            recheck_lot = assign_df['num_lot'].unique()
                            
                            recheck_df = observation.loc[np.logical_and(observation.num_lot.isin(recheck_lot),
                                                                        observation.completion_time == 0)]
                            recheck_df = FitnessCalculation.calculate_finished_time(self, recheck_df)
                            observation.update(recheck_df)
                        else:
                            succeed_observation = observation.loc[np.logical_and(observation.completion_time == 0,
                                                                         observation.index >= row_index)]
                            assign_df = FitnessCalculation.get_different_machine_groupable_df(self, part, coat_design, fabrication_part, operation, precede_observation)
                            assign_df_size = len(assign_df)
                        
                            if(assign_df_size == 1):
                                precede_observation = observation.loc[np.logical_and(observation.completion_time == 0,
                                                                         observation.index <= row_index)]
                                assign_df = FitnessCalculation.get_groupable_df(self, part, coat_design, fabrication_part, operation, machine, precede_observation)
                                assign_df_size = len(assign_df)
                                assign_df_index = assign_df.index.tolist()
                                #print('The leftover')
                                observation.loc[assign_df_index, ['min_assign', 'max_assign']] = assign_df_size
                                observation.at[row_index, ['completion_time',\
                                                         'machine_completion_time',
                                                         'lot_completion_time']] = completion_time
        return observation

    def calculate_weighted_tardiness(self, observation):
        '''Calculate Demand total Weight Tardiness'''
        weighted_tardiness = 0
        demand_weight = self.demand['weight'].tolist()
        lead_time = self.demand['lead_time'].tolist()
        observation = FitnessCalculation.calculate_finished_time(self, observation)
        demand_job_list = observation['num_job'].unique()
        for demand_job in demand_job_list:
            demand_job = int(demand_job)
            demand_job_completion_time = observation.loc[observation.num_job == demand_job]['completion_time'].max()
            if ((lead_time[demand_job] * 24 - demand_job_completion_time) * demand_weight[demand_job]) < 0:
                weighted_tardiness += ((lead_time[demand_job] * 24 - demand_job_completion_time) * demand_weight[demand_job])
        observation.drop(['min_assign', 'max_assign', 'completion_time','machine_completion_time', 'lot_completion_time', 'prev_operation', 'post_operation', 'prev_operation_index', 'post_operation_index' ], axis=1, inplace=True)
        return round(weighted_tardiness, 1)
