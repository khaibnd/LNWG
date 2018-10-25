'''time to go'''
import random
import sta
import pandas as pd
import numpy as np
from copy import deepcopy
from src.initial_solution.initial_solution import InitialSolution
from src.fitness_calculation.fitness_calculation import FitnessCalculation


# pylint: disable=C0103
class ChrosMutation():
    '''1. each child chromosome with small probabilities 
       2. Each single chromosome at the gene level to alter information contained in the gene.'''
    def __itit__(self, population_dict, parameter, sequence, machine):
        self.population_dict = population_dict
        self.parameter = parameter
        self.sequence = sequence
        self.machine = machine
    

    def machine_list(self):
        operation_list = self.machine.columns.tolist()[1:]
        machine_per_operation =  {operation: sta(self.machine[self.machine['criteria']\
                                            == 'num_machine'][operation].values[0])
                                                             for operation in operation_list}
        return sorted(set([i[0] for i in machine_per_operation.values()]))
    def machine_assign_time(self):
        '''Randomly selection machine for operation from list of num_machine'''
        machine_list = ChrosMutation.machine_list(self)
        return {operation: 0 for operation in machine_list}


    def chros_mutation(self):
        '''Make mutation'''
        ROAM = self.parameter[self.parameter.name == 'ROAM']['value'].values[0]
        OSSM = self.parameter[self.parameter.name == 'OSSM']['value'].values[0]
        OSSM_rate = int(self.parameter[self.parameter.name == 'OSSM_rate']['value'].values[0])
        IJMM = self.parameter[self.parameter.name == 'IJMM']['value'].values[0]
        IJMM_rate = int(self.parameter[self.parameter.name == 'IJMM_rate']['value'].values[0])
        ITSM = self.parameter[self.parameter.name == 'ITSM']['value'].values[0]


        for parent_index, observation in iter(self.population_dict.items()):

            # Random Operation Assignment Mutation (ROAM)
            if ROAM > np.random.uniform(0,1):
                observation_size = len(observation)
                df_ROAM_pick = observation.sample(int(observation_size*ROAM),
                                                        replace=False)
                df_ROAM_pick = df_ROAM_pick.sort_index()
                for row_idx, row in df_ROAM_pick.iterrows():
                    row_num_sequence = row['num_sequence']
                    row_operation = row['operation']
                    observation_same_operation = observation.loc[(observation.operation == row_operation)
                                                                 & (observation.num_sequence == row_num_sequence)]
                    observation_same_operation_freq = observation_same_operation['machine'].value_counts()

                    if observation_same_operation_freq.sum() > 0:
                        assigned_machine = observation_same_operation_freq.idxmin()
                        observation.loc[row_idx, 'machine'] = assigned_machine


            # Operations Sequence Shift Mutation (OSSM)
            if OSSM > np.random.uniform(0, 1):
                observation_size = len(observation)
                observation_idx_list = list(range(0,observation_size))
                df_OSSM_pick = random.sample(observation_idx_list, OSSM_rate)
                df_OSSM_pick = sorted(df_OSSM_pick)

                for row_idx in df_OSSM_pick:
                    row = observation[observation.index==row_idx]
                    new_row_idx = random.choice(range(observation.shape[0]))

                    child = pd.DataFrame(columns=['num_job',
                                                  'num_lot',
                                                  'part',
                                                  'operation',
                                                  'machine'])
        
                    # Swap position of OSSM pick to new random position in parent
                    if new_row_idx > row_idx:
                        child = child.append(observation.loc[0:row_idx-1,:], ignore_index=True)
                        child = child.append(observation.loc[row_idx +1:new_row_idx,:], ignore_index=True)
                        child = child.append(row, ignore_index=True)
                        child = child.append(observation.loc[new_row_idx+1:,:], ignore_index=True)
            
                    elif new_row_idx < row_idx:
                        child = child.append(observation.loc[0:new_row_idx-1,:], ignore_index=True)
                        child = child.append(row, ignore_index=True)
                        child = child.append(observation.loc[new_row_idx:row_idx-1,:], ignore_index=True)
                        child = child.append(observation.loc[row_idx+1:,:], ignore_index=True)

                    observation = InitialSolution.observation_sequence_sort(self, child)

            # Inteligent Job Machinr Mutation (IJMM)            
            if IJMM > np.random.uniform(0,1):
                
                IJMM_pick_df = observation.sample(IJMM_rate,replace=False)
                IJMM_pick_df = IJMM_pick_df.sort_index()
                print(IJMM_pick_df)
                for row_idx, row in IJMM_pick_df.iterrows():
                    row_part = row['part']
                    row_num_lot = row['num_lot']
                    row_operation = row['operation']
                    row_machine = row['machine']
                    row_num_sequence = row['num_sequence']
                    prev_row_operation = FitnessCalculation.prev_part_sequence_operation(self, row_part, row_operation, row_num_sequence)
                    post_row_operation = FitnessCalculation.post_part_sequence_operation(self, row_part, row_operation, row_num_sequence)
        
                    try:
                        row_min_idx = observation.index[(observation.num_lot == row_num_lot)
                                                        & (observation.operation == prev_row_operation)].tolist()[0]
                    except:
                        row_min_idx = 0
                    try:
                        row_max_idx = observation.index[(observation.num_lot == row_num_lot)
                                                        & (observation.operation == post_row_operation)].tolist()[0]
                    except:
                        row_max_idx = len(observation)
        
                    check_machine_df = observation.loc[(observation.machine == row_machine)
                                                      & (observation.num_sequence == row_num_sequence)
                                                      & (observation.index > row_min_idx)
                                                      & (observation.index < row_max_idx)]
                    
                    df = pd.DataFrame()
                    for check_row_idx, check_row in check_machine_df.iterrows():
                        check_row_part = check_row['part']
                        check_row_num_sequence = check_row['num_sequence']
                        check_row_num_lot = check_row['num_lot']

                        check_row_operation = check_row['operation']
        
                        check_prev_row_operation = FitnessCalculation.prev_part_sequence_operation(self, check_row_part, check_row_operation, check_row_num_sequence)
                        try:
                            check_prev_row_operation_idx = check_machine_df.index[(observation.num_lot == check_row_num_lot)
                                                        & (observation.operation == check_prev_row_operation)].tolist()[0]
                        except:
                            check_prev_row_operation_idx = None
                        
                        check_post_row_operation = FitnessCalculation.post_part_sequence_operation(self, check_row_part, check_row_operation, check_row_num_sequence)
                        
                        try:
                            check_post_row_operation_idx = check_machine_df.index[(observation.num_lot == check_row_num_lot)
                                                        & (observation.operation == check_post_row_operation)].tolist()[0]
                        except:
                            check_post_row_operation_idx = None
                                        
                        if ((check_prev_row_operation_idx == None) and
                            (check_post_row_operation_idx == None)):
                            df = df.append(check_row)
        
                    if (len(df) >= 2):
                        df_idx = df.index.tolist()
                        df = df.sort_values(['num_job'])
                        df.index = df_idx

                    observation.update(df)


            # Intelligent Task Sort Mutation (ITSM)
            if ITSM > np.random.uniform(0,1):
                part_list = sorted(observation['part'].unique())
                leftover_df = pd.DataFrame()
                temp_df = pd.DataFrame()
                for part in part_list:
                    group_part_job = observation.loc[observation.part == part].copy()
                    num_sequence_list = sorted(group_part_job['num_sequence'].unique())
                    for num_sequence in num_sequence_list:
                        group_num_sequence = group_part_job.loc[group_part_job.num_sequence == num_sequence]
                        packing_index_list = group_num_sequence.index[group_num_sequence.operation == 'packing'].tolist()
                        
                        def quick_sort(PACKING_INDEX_LIST, group_num_sequence):
                            '''Using code from github.com/TheAlgorithms/Python/blob/master/sorts/quick_sort.py'''
                            length = len(PACKING_INDEX_LIST)
                            if length <= 1:
                                return PACKING_INDEX_LIST
                            else:
                                PIVOT = PACKING_INDEX_LIST[0]
                                GREATER = [element for element in PACKING_INDEX_LIST[1:] if group_num_sequence.at[element, 'num_job'] >= group_num_sequence.at[PIVOT, 'num_job']]
                                LESSER = [element for element in PACKING_INDEX_LIST[1:] if group_num_sequence.at[element, 'num_job'] < group_num_sequence.at[PIVOT, 'num_job']]
                                return quick_sort(LESSER, group_num_sequence) + [PIVOT] +quick_sort(GREATER, group_num_sequence)
                            
                        new_packing_index_list = quick_sort(packing_index_list, group_num_sequence)
                        
                        for idx in range(len(packing_index_list)):
                            new_pack_idx = new_packing_index_list[idx]
                            old_pack_idx = packing_index_list[idx]
        
                            print(old_pack_idx, new_pack_idx)
                            
                            old_lot = group_num_sequence.at[old_pack_idx, 'num_lot']
                            old_df = deepcopy(group_num_sequence[group_num_sequence['num_lot'] == old_lot].sort_index())
                            
                            new_lot = group_num_sequence.at[new_pack_idx, 'num_lot']
                            new_df = deepcopy(group_num_sequence[group_num_sequence['num_lot'] == new_lot].sort_index())
                            get_length = min(len(old_df), len(new_df))
                            if len(old_df) >= len(new_df):
                                old_df = old_df[-get_length:]

                                for new_row_idx, new_row in new_df.iterrows():
                                    new_opt = new_row['operation']
                                    old_row_idx = old_df.index[old_df['operation'] == new_opt].tolist()[0]
                                    df = pd.DataFrame({old_row_idx : new_row})
                                    df = df.T
                                    temp_df = temp_df.append(df)
                            
                            else:
                                leftover_df = leftover_df.append(new_df[:- get_length])
                                new_df = new_df[-get_length:]

                                for new_row_idx, new_row in new_df.iterrows():
                                    new_opt = new_row['operation']
                                    old_row_idx = old_df.index[old_df['operation'] == new_opt].tolist()[0]
                                    df = pd.DataFrame({old_row_idx : new_row})
                                    df = df.T
                                    temp_df = temp_df.append(df)

                temp_full = temp_df.append(leftover_df)
                temp_full = temp_full.sort_index()
                temp_full = temp_full.reset_index(drop=True)
                
            self.population_dict[parent_index] = temp_full

        return self.population_dict

