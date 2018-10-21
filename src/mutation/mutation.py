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
        population_size = self.parameter[self.parameter.name == 'population_size']['value']
        ROAM = self.parameter[self.parameter.name == 'ROAM']['value'].values[0]
        IOAM = self.parameter[self.parameter.name == 'IOAM']['value'].values[0]
        OSSM = self.parameter[self.parameter.name == 'OSSM']['value'].values[0]
        for parent_index, observation in iter(self.population_dict.items()):
            
            # RHS Random Operation Assignment Mutation (ROAM)
            if ROAM > np.random.uniform(0, 1):

                df_ROAM_pick = observation.sample(int(population_size*ROAM),
                                               replace=False)
                for index, row in df_ROAM_pick.iterrows():
                    num_sequence = row['num_sequence']
                    operation = row['operation']
                    machine_list = InitialSolution.generate_machine_list(self, num_sequence, operation)

                # Replace different machine to the row
                    self.population_dict[parent_index].loc[index, 'machine'] = random.choice(machine_list)
        

            # Operations Sequence Shift Mutation (OSSM)
            if OSSM > np.random.uniform(0, 1):
                OSSM_pick = observation.sample(n=1)
                new_position_index = random.choice(range(observation.shape[0]))
                child = pd.DataFrame(columns=['num_job',
                                              'num_lot',
                                              'part',
                                              'operation',
                                              'machine'])
                # Swap position of OSSM pick to new random position in parent
                if new_position_index > OSSM_pick.index.values[0]:
                    child = child.append(observation.loc[:OSSM_pick.index.values[0]-1,:], ignore_index=True)
                    child = child.append(observation.loc[OSSM_pick.index.values[0]+1:new_position_index,:], ignore_index=True)
                    child = child.append(OSSM_pick, ignore_index=True)
                    child = child.append(observation.loc[new_position_index+1:,:], ignore_index=True)

                elif new_position_index < OSSM_pick.index.values[0]:
                    child = child.append(observation.loc[0:new_position_index-1,:], ignore_index=True)
                    child = child.append(OSSM_pick, ignore_index=True)
                    child = child.append(observation.loc[new_position_index:OSSM_pick.index.values[0]-1,:], ignore_index=True)
                    child = child.append(observation.loc[OSSM_pick.index.values[0]+1:,:], ignore_index=True)
                else:
                    child = observation
                    
                observation = child
                
            observation = InitialSolution.observation_sequence_sort(self, observation)
            self.population_dict[parent_index] = observation


            # Intelligent Task Sort Mutation (IOAM)
            # Chuyen cac taskcan lam xong som len truoc
            if IOAM > np.random.uniform(0,1):
                part_list = sorted(observation['part'].unique())
                for part in part_list:
                    group_part_job = observation.loc[observation.part == part]
                    num_sequence_list = sorted(group_part_job['num_sequence'].unique())
                    temp_df = pd.DataFrame()
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
                        temp_df = pd.DataFrame()

                        for pack_idx in packing_index_list:
                            new_pack_idx = new_packing_index_list.index(pack_idx)
                            new_pack_idx = packing_index_list[new_pack_idx]

                            old_lot = group_num_sequence.at[pack_idx, 'num_lot']
                            old_df = group_num_sequence[group_num_sequence.num_lot == old_lot].copy().sort_index()
                            
                            new_lot = group_num_sequence.at[new_pack_idx, 'num_lot']
                            new_df = group_num_sequence[group_num_sequence.num_lot == new_lot].copy().sort_index()
                            
                            get_length = min(len(old_df), len(new_df))
                            if len(old_df) == get_length:
                                old_df = old_df[-get_length:]
                            else:
                                new_df = new_df[-get_length:]
        
                            for fwd_row_idx, fwd_row in old_df.iterrows():
                                for bwd_row_idx, bwd_row in new_df.iterrows():
                                    if fwd_row['operation'] == bwd_row['operation']:
                                        fwd_row['start_time'] = bwd_row['start_time']
                                        fwd_row['completion_time'] = bwd_row['completion_time']
                                        new_df = pd.DataFrame({bwd_row_idx : fwd_row})
                                        new_df = new_df.T
                                        temp_df = temp_df.append(new_df)
                        
                        # print('temp', temp_df)
        
                    observation.update(temp_df)


        return self.population_dict