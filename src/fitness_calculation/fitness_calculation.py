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
        '''Dertermine previous operation of current gene in its operation'''
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
        '''Dertermine next operation of current gene in its operation'''
        part_sequence = InitialSolution.part_sequence(self, part, num_sequence)
        operation_index = part_sequence.index(operation)
        try:
            post_part_sequence_operation = part_sequence[operation_index + 1]
        except:
            post_part_sequence_operation = None
        return post_part_sequence_operation
    
    
    def num_lot_operation_index(self, num_lot, operation, chromos):
        '''Get gene index inside chromosome base on num_lot and operation'''
        try:
            index = chromos.index[np.logical_and(chromos.num_lot == num_lot,
                                                 chromos.operation == operation)].tolist()
            index = index[0]
        except:                                 
            index = None
        return index
        
    
    def calculate_job_processing_time(self, part, operation, num_sequence):
        '''VLOOKUP gene's completion time on database: self.processing_time'''
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

 
    def calculate_job_processing_time_plus(self, job_processing_time, row, chromos):
        '''Adding daily machine none workign hours to the job competion time'''
        machine = row['machine']
        num_lot = row['num_lot']
        machine_daily_work_hour = self.machine_criteria.loc[self.machine_criteria.machine == machine, 'work_hour'].values[0]
        machine_completion_time = chromos.loc[chromos.machine == machine][ 'machine_completion_time'].max()
        lot_completion_time = chromos.loc[chromos.num_lot == num_lot]['lot_completion_time'].max()
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
        '''
        Filter an Dataframe and get new one with condition ALL ON SAME MACHINE
        to combine into 1 run lot:

        1. Similar coat design on coat operation
        2. Similar coat design on block wedge operation, that support coat operation
        3. Similar raw fabrication part on rob slicing operation
        4. Similar part on other operations
        '''
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
        '''
        Filter an Dataframe and get new one with condition ON DIFFERENT MACHINES
        to combine into 1 run lot:

        1. Similar coat design on coat operation
        2. Similar coat design on block wedge operation, that support coat operation
        3. Similar raw fabrication part on rob slicing operation
        4. Similar part on other operations
        '''
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


    def completion_time_assign_condition(self, row_index, row, chromos):
        '''
        Check condition to calculate gene's completion time:
        1. The completion_time has yet assigned
            - The first operation of lot.
            - Or the its previous operation' s competion_time has assigned
            - Or the index of row not in chros index (recalculate completion_time)
        '''
        chromos_index_list = chromos.index.tolist()

        if (np.logical_and(chromos.at[row_index, 'completion_time'] == 0,
                           np.logical_or(len(chromos) == 531,
                                         chromos.at[row_index, 'review'] == 1))):
            if pd.isnull(row['prev_operation_index']):
                return True
            elif (int(row['prev_operation_index'])not in (chromos_index_list)):
                return True
            elif chromos.at[int(row['prev_operation_index']), 'completion_time'] > 0:
                return True


    def calculate_completion_time(self, observation_list, observation):
        '''
        Get completion time of the gene:
        1. Input original chromosome
        2. Adding extra gene code to the gene, that in order to support pandas calculation (extra column)
             - min_assign: originally equal 0
             - max_assign: originally equal 0
             - completion_time: originally equal 0
             - machine_completion_time: originally equal 0
             - lot_completion_time: originally equal 0
             - coat_design: VLOOKUP from database: self.criteria
             - fabrication_part: VLOOKUP from database: self.criteria
             - prev_operation: originally equal None
             - post_operation: originally equal None
             - prev_operation_index: originally equal None
             - post_operation_index: originally equal None
        2. Iterate each gene in chromosome:
            2.1. IF the gene satisfy assign condition then move to the next run, else move to the next gene
                Get gene code value of the gene: part, num_lot, operation...
                Calculate completion_time
                2.1.1. IF gene lotsize equal 1:
                    - Assign gene completion_time & machine_completion_time &
                                                        lot_completion_time = completion_time
                    - Assign min_assign & max_assign = 1
                2.1.2. IF gene min_lotsize equal 1, max_lotsize greater than 1:
                    - Get succeed gene DF
                    - Filter the succeed gene DF ussing FILTER FUNC to gene assign DF
                        describe number of lots that can run together
                    - Slice the size assign DF to the limit of max_lotsize
                    - Calculate the gene completion_time
                    - Assign all assign DF completion_time & machine_completion_time &
                                                                lot_completion_time = completion_time
                    - Assign all assign DF min_assign  = 1
                    - Assign all assign DF max_assign = grouable DF's size
                2.1.3. IF gene min_lotsize greater than 1, max_lotsize equal min_lotsize:
                    - Get precede gene DF that completion_time has yet assigned
                    - Get assign DF with same machine on precede gene DF
                    - Calculate lenghth of assign DF
                    2.1.3.1 IF lenghth of assign DF equal min_lotsize:
                        - Get list gene index of assign DF
                        - Assign all assign DF completion_time & machine_completion_time &
                                                                lot_completion_time = completion_time
                        - Assign all assign DF min_assign & max_assign  = min_lotsize
                        - Get list of num_lot of assign DF
                        - Get check DF that contain all the genes of above num_lot list and completion_time has yet assign
                        - Recusive call completion_time_calculation for this check DF
                        - Base on the new check DF, update the gene code of original chromosone's affected genes
                    2.1.3.2.ELSE:
                        - Get assign DF with different machine on precede gene DF
                        2.1.3.2.1. IF lenghth of assign DF equal 2 * min_lotsize:
                            - Slice the size assign DF to the limit of max_lotsize 
                            - Get list gene index of assign DF
                            - Assign all assign DF completion_time & machine_completion_time &
                                                                    lot_completion_time = completion_time
                            - Assign all assign DF min_assign & max_assign  = min_lotsize
                            - Get list of num_lot of assign DF
                            - Get check DF that contain all the genes of above num_lot list and
                                                                            completion_time has yet assign
                            - Recusive call completion_time_calculation for this check DF
                            - Base on the new check DF, update the gene code of original chromosone's affected genes
                        2.1.3.2.2. ELSE:
                        - Get succeed gene DF that completion_time has yet assigned
                        - Get assign DF with same machine on precede gene DF
                        - Get list gene index of assign DF
                        - Assign all assign DF completion_time & machine_completion_time &
                                                                lot_completion_time = completion_time
                        - Assign all assign DF min_assign & max_assign  = min_lotsize
                        - Get list of num_lot of assign DF
                        - Get check DF that contain all the genes of above num_lot list and completion_time has yet assign
                        - Recusive call completion_time_calculation for this check DF
                        - Base on the new check DF, update the gene code of original chromosone's affected genes
                2.1.3. IF gene min_lotsize greater than 1, max_lotsize greater than min_lotsize:
                    (Not consider currently)
        
        3. Return chromosome with new value of gene code
        '''
        #observation_list = observation.loc[observation.completion_time == 0].index.tolist()
        
        print('observation len', len(observation))
        #print('row index 531 value completion time', observation.at[530, 'completion_time'])
        for row_index in observation_list:
            observation.loc[row_index, 'review'] += 1
            '''Start main calculation'''
            print('row_index', row_index)
            # Check the job yet observe
            row = observation.loc[row_index]
            if (FitnessCalculation.completion_time_assign_condition(self, row_index, row, observation)):
                part = observation.at[row_index, 'part']
                operation = observation.at[row_index, 'operation']
                machine = observation.at[row_index, 'machine']
                num_sequence = observation.at[row_index, 'num_sequence']
                coat_design = observation.at[row_index, 'coat_design']
                fabrication_part = observation.at[row_index, 'fabrication_part']
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
                completion_time = FitnessCalculation.calculate_job_processing_time_plus(self,
                                                                                        job_processing_time,
                                                                                        row, observation)
                # Step 6.1: Machine load lotsize = 1
                if (np.logical_and(max_lotsize == 1, min_lotsize == 1)):
                    observation.at[row_index, ['completion_time',\
                                               'machine_completion_time',\
                                               'lot_completion_time']] = completion_time
                    observation.at[row_index, ['min_assign', 'max_assign']] = 1

                # Step 6.2: maximum lotsize > 1 and minimum lotsize = 1
                elif (np.logical_and(max_lotsize > 1, min_lotsize == 1)):
                    succeed_observation = observation.loc[row_index:, :]
                    groupable_df = FitnessCalculation.get_groupable_df(self,
                                                                       part, coat_design, fabrication_part, operation,
                                                                       machine, succeed_observation)

                    #The group index is first operation of sequence
                    if (pd.isnull(observation.at[row_index, 'prev_operation_index'])):
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
                                                     'machine_completion_time',\
                                                     'lot_completion_time']] = completion_time
                
                # Step 6.3: minimum lotsize = minimum lotsize > 1
                elif (np.logical_and(min_lotsize == max_lotsize, min_lotsize > 1)):
                    #print('condition 3')
                    precede_observation = observation.loc[np.logical_and(observation.completion_time == 0,
                                                                         observation.index <= row_index)]
                    assign_df = FitnessCalculation.get_groupable_df(self, part, coat_design, fabrication_part, operation, machine, precede_observation)
                    assign_df_size = len(assign_df)
                    # Previous contain enough lot for minimum lotsize load
                    if assign_df_size >= min_lotsize:
                        #print('min full assign same machine')
                        print('aaaaaaaaaaaaaaaaa')
                        assign_df = assign_df[:min_lotsize]
                        assign_df_index = assign_df.index.tolist()
                        observation.loc[assign_df_index, ['min_assign', 'max_assign']] = min_lotsize
                        #print(min_lotsize)
                        #print(observation.loc[min_lotsize_index,['min_assign', 'max_assign']])
                        # Con thieu dieu kien can max_lotsize phai la max hien tai va cai moi
                        observation.at[assign_df_index, ['completion_time',\
                                                         'machine_completion_time',\
                                                         'lot_completion_time']] = completion_time
                        #calculate
                        
                        check_df_lot = assign_df['num_lot'].unique()
                        check_df = precede_observation.loc[np.logical_and(precede_observation.num_lot.isin(check_df_lot),
                                                                    precede_observation.completion_time == 0)]
                        
                        check_df_list = check_df.index.tolist()
                        observation = FitnessCalculation.calculate_completion_time(self, check_df_list, observation)
                    
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
                                                             'machine_completion_time',\
                                                             'lot_completion_time']] = completion_time
                            
                            #calculate
                            
                            check_df_lot = assign_df['num_lot'].unique()
                            check_df = precede_observation.loc[np.logical_and(precede_observation.num_lot.isin(check_df_lot),
                                                                    precede_observation.completion_time == 0)]
                            check_df_list = check_df.index.tolist()
                            observation = FitnessCalculation.calculate_completion_time(self, check_df_list, observation)
                        
                        else:
                            print(row_index)
                            print(len(observation_list))
                            if len(observation_list)==531:
                                succeed_observation = observation.loc[np.logical_and(observation.completion_time == 0,
                                                                             observation.index >= row_index)]
                                assign_df = FitnessCalculation.get_different_machine_groupable_df(self, part, coat_design, fabrication_part, operation, succeed_observation)
                                assign_df_size = len(assign_df)
                            
                                if(assign_df_size == 1):
                                    precede_observation = observation.loc[observation.completion_time == 0]
                                    assign_df = FitnessCalculation.get_groupable_df(self, part, coat_design, fabrication_part, operation, machine, precede_observation)
                                    assign_df_size = len(assign_df)
                                    assign_df_index = assign_df.index.tolist()
                                    print('The leftover')
                                    observation.loc[assign_df_index, ['min_assign', 'max_assign']] = assign_df_size
                                    observation.at[assign_df_index, ['completion_time',\
                                                                     'machine_completion_time',\
                                                                     'lot_completion_time']] = completion_time
                                    #calculate
                                    check_df_lot = assign_df['num_lot'].unique()
                                    check_df = precede_observation.loc[np.logical_and(precede_observation.num_lot.isin(check_df_lot),
                                                                            precede_observation.completion_time == 0)]
                                    check_df_list = check_df.index.tolist()
                                    observation = FitnessCalculation.calculate_completion_time(self, check_df_list, observation)
                                
        return observation
    
    
    def calculate_weighted_tardiness(self, observation):
        '''Calculate Demand total Weight Tardiness'''
        observation_size = len(observation)
        observation['review'] = pd.Series([0] * observation_size)
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
        weighted_tardiness = 0
        demand_weight = self.demand['weight'].tolist()
        lead_time = self.demand['lead_time'].tolist()
        
        writer1 = pd.ExcelWriter(r'/Users/khaibnd/github-repositories/LNWG/src/data/new_output1.xlsx')
        observation.to_excel(writer1, sheet_name='aaa')
        writer1.save()
        observation = FitnessCalculation.calculate_completion_time(self, observation.index.tolist(), observation)
        
        writer = pd.ExcelWriter(r'/Users/khaibnd/github-repositories/LNWG/src/data/new_output2.xlsx')
        observation.to_excel(writer, sheet_name='aaa')
        writer.save()
        print('len fitness', len(observation))
        
        demand_job_list = observation['num_job'].unique()
        for demand_job in demand_job_list:
            demand_job = int(demand_job)
            demand_job_completion_time = observation.loc[observation.num_job == demand_job]['completion_time'].max()
            if ((lead_time[demand_job] * 24 - demand_job_completion_time) * demand_weight[demand_job]) < 0:
                weighted_tardiness += ((lead_time[demand_job] * 24 - demand_job_completion_time) * demand_weight[demand_job])
        observation.drop(['min_assign', 'max_assign', 'completion_time','machine_completion_time', 'lot_completion_time', 'prev_operation', 'post_operation', 'prev_operation_index', 'post_operation_index' ], axis=1, inplace=True)
        return round(weighted_tardiness, 1)
