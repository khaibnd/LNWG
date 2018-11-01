'''Calculate production leadtime per observation'''
import math
import numpy as np
import pandas as pd
from src.initial_solution.initial_solution import InitialSolution


class FitnessCalculation():
    '''Measure invidial demand job completion time and Total Weighed Tardiness'''

    def __init__(self, observation, sequence, machine_lotsize):
        self.observation = observation
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

    def calculate_job_processing_time(self, part, operation, num_sequence):
        '''Completion time each num_lot(= batch processing time)'''

        def line_index(num_sequence, part):
            line_index = self.processing_time.index[(self.processing_time['num_sequence'] == num_sequence)
                                              & (self.processing_time['part'] == part)].tolist()
            # Get the smallest index
            line_index = line_index[0]
            return line_index

        # print('line', part, operation, num_sequence)
        job_processing_time = self.processing_time.at[line_index(num_sequence, part), operation]
        # print('job_processing_time', part, operation, num_sequence, job_processing_time)
        return job_processing_time

    def completion_time_with_daily_work_hour(self,
                                             job_processing_time,
                                             row_index,
                                             machine,
                                             observation):
        machine_daily_work_hour = self.machine_criteria.loc[self.machine_criteria.machine == machine, 'work_hour'].values[0]
        num_day = math.floor((observation.at[row_index, 'completion_time'] - observation.at[row_index, 'machine_completion_time']) / machine_daily_work_hour)
        free_hour = observation.at[row_index, 'completion_time'] - observation.at[row_index, 'machine_completion_time'] - job_processing_time
        if num_day > 0 and free_hour == 0:
            observation.at[row_index, 'machine_completion_time'] = observation.at[row_index, 'completion_time'] + num_day * (24 - machine_daily_work_hour)
        elif num_day > 0 and free_hour <= num_day * (24 - machine_daily_work_hour):
            observation.at[row_index, 'machine_completion_time'] = observation.at[row_index, 'completion_time'] + num_day * (24 - machine_daily_work_hour) - free_hour
        else:
            observation.at[row_index, 'machine_completion_time'] = observation.at[row_index, 'completion_time']
        return observation.at[row_index, 'machine_completion_time']
    
    def get_machine_completion_time(self, observation, index, machine):

        machine_competion_time_series = observation[np.logical_and(observation.index <= index,
                                                        observation.machine == machine)].max()
        machine_competion_time = machine_competion_time_series.get('machine_completion_time')

        return machine_competion_time


    def get_lot_completion_time(self, observation, index, num_lot):
        lot_competion_time_series= observation[np.logical_and(observation.index <= index,
                                                        observation.num_lot == num_lot)].max()
        lot_competion_time = lot_competion_time_series.get('lot_completion_time')
        return lot_competion_time
    
    
    def get_demand_job_completion_time(self, demand_job, observation):
        demand_job_completion_time = observation[(observation.num_job == demand_job), 'num_job'].max()
        return demand_job_completion_time
    

    def ga_calculation(self, job_processing_time, row_index, num_lot, machine, observation):
        observation.at[row_index, 'machine_completion_time'] = FitnessCalculation.get_machine_completion_time(self,
                                                                                                              observation,
                                                                                                              row_index,
                                                                                                              machine)
        
        observation.at[row_index, 'lot_completion_time'] = FitnessCalculation.get_lot_completion_time(self,
                                                                                                      observation,
                                                                                                      row_index,
                                                                                                      num_lot)
        # Step 5.1: First assignment to machine m and operation                                                                                                  num_lot)
        if (np.logical_and(observation.at[row_index, 'machine_completion_time'] == 0,
                           observation.at[row_index, 'lot_completion_time'] == 0)):
            print('row_index',row_index, job_processing_time)
            observation.at[row_index, 'completion_time'] = job_processing_time
            observation.at[row_index, 'lot_completion_time'] = job_processing_time
            observation.at[row_index, 'machine_completion_time'] = FitnessCalculation.completion_time_with_daily_work_hour(self,
                                                                                                                           job_processing_time,
                                                                                                                           row_index,
                                                                                                                           machine,
                                                                                                                           observation)
        # Step 5.2: First assignment to machine m and operation
        elif (np.logical_and(observation.at[row_index, 'machine_completion_time'] == 0,
                           observation.at[row_index, 'lot_completion_time'] > 0)):
            observation.at[row_index, 'completion_time'] = observation.at[row_index, 'lot_completion_time'] + job_processing_time
            observation.at[row_index, 'lot_completion_time'] = observation.at[row_index, 'completion_time']
            observation.at[row_index, 'machine_completion_time'] = FitnessCalculation.completion_time_with_daily_work_hour(self,
                                                                                                                           job_processing_time,
                                                                                                                           row_index,
                                                                                                                           machine,
                                                                                                                           observation)
        # Step 5.3: First operation in sequence, and machine m assignment > 1
        elif (np.logical_and(observation.at[row_index, 'machine_completion_time'] > 0,
                           observation.at[row_index, 'lot_completion_time'] == 0)):
            observation.at[row_index, 'completion_time'] = observation.at[row_index, 'machine_completion_time'] + job_processing_time
            observation.at[row_index, 'lot_completion_time'] = observation.at[row_index, 'completion_time']
            observation.at[row_index, 'machine_completion_time'] = FitnessCalculation.completion_time_with_daily_work_hour(self,
                                                                                                                           job_processing_time,
                                                                                                                           row_index,
                                                                                                                           machine,
                                                                                                                           observation)
        # Step 5.4: Not the first operation, and machine m assignment > 1
        else:
            observation.at[row_index, 'completion_time'] = max(observation.at[row_index, 'lot_completion_time'],
                                                               observation.at[row_index, 'machine_completion_time']) + job_processing_time
            observation.at[row_index, 'lot_completion_time'] = observation.at[row_index, 'completion_time']
            observation.at[row_index, 'machine_completion_time'] = FitnessCalculation.completion_time_with_daily_work_hour(self,
                                                                                                                           job_processing_time,
                                                                                                                           row_index,
                                                                                                                           machine,
                                                                                                                           observation)
        return observation
  
    def line_index(self, machine, num_sequence):
        try:
            line_index = self.machine_lotsize.index[(self.machine_lotsize.machine == machine)
                                                      & (self.machine_lotsize.num_sequence == num_sequence)].tolist()
            line_index = line_index[0]
            
        except(IndexError):
            print('IndexError: list index out of range')
            print('Combination %s and %s is not in the input file' % (machine, num_sequence))
        return line_index

    def calculate_finished_time(self, observation):
        '''Chua xem xet may tsk(Gop 4 lot)'''
        observation_size = len(observation)

        observation['min_assign'] = pd.Series([0] * observation_size)
        observation['max_assign'] = pd.Series([0] * observation_size)
        observation['completion_time'] = pd.Series([0.00] * observation_size)
        observation['machine_completion_time'] = pd.Series([0.00] * observation_size)
        observation['lot_completion_time'] = pd.Series([0.00] * observation_size)
        pending_job_index = []
        for row_index, row in observation.iterrows():
            num_lot = row['num_lot']
            part = row['part']
            operation = row['operation']
            machine = row['machine']
            num_sequence = row['num_sequence']
            try:
                job_processing_time = FitnessCalculation.calculate_job_processing_time(self,
                                                                                       part,
                                                                                       operation,
                                                                                       num_sequence)
            except:
                print('Wrong processing time!')
                print('part: %s, operation: %s, num_sequence: %s' % (part, operation, num_sequence))
            

            #print('lot_completion_time', FitnessCalculation.get_machine_completion_time(self, observation, row_index, machine))
            
            def check_groupable_condition(self):
                groupable_condition = False
                if (group_row['operation'] == 'coat' and
                    group_row['machine'] == machine):
                    coat_design = self.criteria.loc[self.criteria.part == part, 'coat_design'].values[0]
                    group_part = group_row['part']
                    group_coat_design = self.criteria.loc[self.criteria.part == group_part, 'coat_design'].values[0]

                    if group_coat_design == coat_design:
                        groupable_condition = True
                        return groupable_condition
                
                if (group_row['operation'] == 'block_wedge' and
                    group_row['machine'] == machine):
                    coat_design = self.criteria.loc[self.criteria.part == part, 'coat_design'].values[0]
                    group_part = group_row['part']
                    group_coat_design = self.criteria.loc[self.criteria.part == group_part, 'coat_design'].values[0]

                    if group_coat_design == coat_design:
                        groupable_condition = True
                        return groupable_condition
                
                
                
                elif (group_row['operation'] == 'rob_slicing' and
                    group_row['machine'] == machine):
                    fabrication_part = self.criteria.loc[self.criteria.part == part, 'fabrication_part'].values[0]
                    group_part = group_row['part']
                    group_fabrication_part = self.criteria.loc[self.criteria.part == group_part, 'fabrication_part'].values[0]

                    if group_fabrication_part == fabrication_part:
                        groupable_condition = True
                        return groupable_condition
                    
                    
                elif (group_row['operation'] != 'rob_slicing' and
                      group_row['operation'] != 'coat' and
                      group_row['part'] == part and
                      group_row['operation'] == operation and
                      group_row['machine'] == machine and
                      group_row['num_sequence'] == num_sequence):
                    groupable_condition = True
                    return groupable_condition
                else:
                    return groupable_condition


            max_lotsize = int(self.machine_lotsize.at[FitnessCalculation.line_index(self, machine, num_sequence), 'max_lotsize'])
            min_lotsize = int(self.machine_lotsize.at[FitnessCalculation.line_index(self, machine, num_sequence), 'min_lotsize'])

            '''Start main calculation'''
            
            pending_job_index = []
            pending_lot = []
            # Step 6.1: Maximum and minimum lot per run = 1
            if (np.logical_and(np.logical_and(max_lotsize == 1,
                                              min_lotsize == 1),
                               num_lot not in pending_lot)):
                FitnessCalculation.ga_calculation(self, job_processing_time, row_index, num_lot, machine, observation)
                observation.at[row_index, 'max_assign'] += 1
                observation.at[row_index, 'min_assign'] += 1


            # Step 6.2: maximum lotsize > 1 and minimum lotsize = 1
            elif (np.logical_and(np.logical_and(min_lotsize == 1, observation.at[row_index, 'max_assign'] == 0),
                                 num_lot not in pending_lot)):
                lot_size = 1
                FitnessCalculation.ga_calculation(self, job_processing_time, row_index, num_lot, machine, observation)
                succeed_observation = observation.loc[row_index + 1:, :]

                # Review groupability of precedence genes
                for group_row_index, group_row in succeed_observation.iterrows():
                    # Step 6.2.1 : Check total lot per run in machine should lower than maximum.
                    if lot_size < max_lotsize:

                        if (check_groupable_condition(self) == True):
                            prev_operation = FitnessCalculation.prev_part_sequence_operation(self,
                                                                                             part,
                                                                                             operation,
                                                                                             num_sequence)
                            # Step 6.2.1.2.1: The group index is first operation of sequence
                            if prev_operation == None:
                                print('First operation')
                                group_row_num_lot = group_row['num_lot']
                                observation.at[group_row_index, 'completion_time'] = observation.at[row_index, 'completion_time']
                                observation.at[group_row_index, 'lot_completion_time'] = observation.at[row_index, 'lot_completion_time']
                                observation.at[group_row_index, 'machine_completion_time'] = observation.at[row_index, 'machine_completion_time']
                                lot_size += 1
                                observation.at[group_row_index, 'max_assign'] += 1

                            # Step 6.2.1.2.1: The group index is not first operation of sequence    
                            else:
                                prev_group_observation = observation.loc[:row_index - 1, :]
                                try:
                                    group_row_num_lot = group_row['num_lot']
                                    prev_row_index = prev_group_observation.index[(prev_group_observation['num_lot'] == group_row_num_lot)
                                                                        & (prev_group_observation['operation'] == prev_operation)].tolist()[0]
                                except:
                                    prev_row_index = None

                                # Step 6.2.1.2.1.1: Previous operation of observation exists
                                if prev_row_index != None:

                                    # Step 6.2.1.2.1.1.1: Job of Previous operation finished BEFORE group job start
                                    if observation.at[prev_row_index, 'completion_time'] < observation.at[row_index, 'completion_time']:
                                        group_row_num_lot = group_row['num_lot']
                                        observation.at[group_row_index, 'completion_time'] = observation.at[row_index, 'completion_time']
                                        observation.at[group_row_index, 'lot_completion_time'] = observation.at[row_index, 'lot_completion_time']
                                        observation.at[group_row_index, 'machine_completion_time'] = observation.at[row_index, 'machine_completion_time']
                                        lot_size += 1
                                        observation.at[group_row_index, 'max_assign'] += 1

                                    # Step 6.2.1.2.1.1.1: Job of Previous operation finished AFTER group job start
                                    else:
                                        try:
                                            group_row_num_lot = group_row['num_lot']
                                            prev_row_index = observation.index[(observation['num_lot'] == group_row_num_lot)
                                                                                & (observation['operation'] == prev_operation)].tolist()[0]
                                        except:
                                            prev_row_index = None
        
                                        # Step 6.2.1.2.1.1: Previous operation of observation exists
                                        if prev_row_index == None:
        
                                            # Step 6.2.1.2.1.1.1: Job of Previous operation finished BEFORE group job start
                                            if observation.at[prev_row_index, 'completion_time'] < observation.at[row_index, 'completion_time']:
                                                group_row_num_lot = group_row['num_lot']
                                                observation.at[group_row_index, 'completion_time'] = observation.at[row_index, 'completion_time']
                                                observation.at[group_row_index, 'lot_completion_time'] = observation.at[row_index, 'lot_completion_time']
                                                observation.at[group_row_index, 'machine_completion_time'] = observation.at[row_index, 'machine_completion_time']
                                                lot_size += 1
                                                observation.at[group_row_index, 'max_assign'] += 1
                                            pass
                                        pass
                                pass
                        
                        # Step 6.2.1.3: Operation canot group together
                        pass
                    # Step 6.2.1 : Total lot per run in machine over maximum.
                    pass

            # Step 6.3: maximum lotsize = 1 and minimum lotsize > 1
            elif (np.logical_and(observation.at[row_index, 'min_assign'] < min_lotsize,
                                 num_lot not in pending_lot)):
                print('min lot size > 1')
                precede_observation = observation.loc[:row_index - 1, :]
                succeed_observation = observation.loc[row_index + 1:, :]

                groupable_precede_observation_df = precede_observation[(precede_observation.index < row_index)&
                                                                       (precede_observation.part == part)&
                                                                       (precede_observation.operation == operation)&
                                                                       (precede_observation.machine == machine)&
                                                                       (precede_observation.num_sequence == num_sequence)&
                                                                       (precede_observation.min_assign < min_lotsize)]

                num_min_lotsize_precede = len(groupable_precede_observation_df)
                
                # Previous contain enough lot for minimum lotsize load
                if num_min_lotsize_precede >= min_lotsize:
                    FitnessCalculation.ga_calculation(self, job_processing_time, row_index, num_lot, machine, observation)

                    min_lotsize_index = groupable_precede_observation_df.index
                    observation.loc[min_lotsize_index,['min_assign', 'max_assign']] = num_min_lotsize_precede
                    # Con thieu dieu kien can max_lotsize phai la max hien tai va cai moi
                    observation.at[min_lotsize_index, 'completion_time'] = observation.at[row_index, 'completion_time']
                    observation.at[min_lotsize_index, 'lot_completion_time'] = observation.at[row_index, 'lot_completion_time']
                    observation.at[min_lotsize_index, 'machine_completion_time'] = observation.at[row_index, 'machine_completion_time']
                    
                    # Get list of lot can re-calculate completion time
                    lot_remover = groupable_precede_observation_df['num_lot'].unique()
                    pending_lot = list(set(pending_lot).difference(set(lot_remover)))
                    
                    recalculate_df = observation.loc[np.logical_and(np.logical_and(observation.num_lot.isin(lot_remover),
                                                                                   observation.index < row_index),
                                                                    observation.completion_time ==0)]
                    # Calculate the competion time
                    recalculate_df = FitnessCalculation.calculate_finished_time(self, recalculate_df)
                    # update to main
                    observation.update(recalculate_df)
                
                # Review the succeed genes in chros
                else:
                    shortage_min_lotsize = min_lotsize - num_min_lotsize_precede
                    groupable_succeed_observation_df = succeed_observation[(succeed_observation.index < row_index)&
                                                                       (succeed_observation.part == part)&
                                                                       (succeed_observation.operation == operation)&
                                                                       (succeed_observation.machine == machine)&
                                                                       (succeed_observation.num_sequence == num_sequence)&
                                                                       (succeed_observation.min_assign < min_lotsize)]

                    num_min_lotsize_succeed =  len(groupable_succeed_observation_df)
                    
                    # Succeed lot size enough to cover 
                    if num_min_lotsize_succeed >= shortage_min_lotsize:
                        groupable_succeed_observation_df = groupable_precede_observation_df.head(shortage_min_lotsize)
                        
                    groupable_precede_observation_df = groupable_precede_observation_df.append(groupable_succeed_observation_df)
                    lot_remover = groupable_precede_observation_df['num_lot'].unique()
                    
                    pending_lot = list(set(pending_lot).difference(set(lot_remover)))
                    
                    recalculate_df = observation.loc[np.logical_and(np.logical_and(observation.num_lot.isin(lot_remover),
                                                                                   observation.index < row_index),
                                                                    observation.completion_time ==0)]
                    # Calculate the competion time
                    recalculate_df = FitnessCalculation.calculate_finished_time(self, recalculate_df)
                    # update to main
                    observation.update(recalculate_df)
                        
                    
                        
                    
                        
            # Step 6.4: maximum lotsize > 1 and minimum lotsize > 1    
            
            # Step 6.5: Cannot group together
            else:
                pending_lot = pending_lot.append(num_lot)

        writer = pd.ExcelWriter(r'/Users/khaibnd/github-repositories/LNWG/src/data/new_output2.xlsx')
        observation.to_excel(writer, sheet_name='output')
        writer.save()
        
        return observation

    def calculate_makespan(self, observation):
        '''Calculate Demand total makespan'''
        prev_operation_completion_time, _, __ = FitnessCalculation.calculate_finished_time(self, observation)
        makespan = max(prev_operation_completion_time.values())
        return round(makespan, 1)

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
        observation.drop(['min_assign', 'max_assign', 'completion_time'], axis=1, inplace=True)
        return round(weighted_tardiness, 1)
