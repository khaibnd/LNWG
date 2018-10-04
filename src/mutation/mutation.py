'''time to go'''
import random
import sta
import pandas as pd
import numpy as np
from copy import deepcopy
from src.initial_solution.initial_solution import InitialSolution



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

            # Intelligent Operations Assignment Mutation (IOAM)
            if IOAM > np.random.uniform(0, 1):
                machine_assign_time = ChrosMutation.machine_assign_time(self)
                for row_index, row in observation.iterrows():
                    machine = row['machine']
                    machine_assign_time[machine] += 1
                for machine_index, assign_time in machine_assign_time.items():
                    max_mc, freq_max = max(list(enumerate(machine_assign_time[n])),
                                           key=lambda x: x[1])
                    min_mc, freq_min = min(list(enumerate(machine_assign_time[n])),
                                           key=lambda x: x[1])
                    # Change machine from maximum workload to the minimum workload
                    total_max_index = []
                    total_min_index = []
                    for i, __ in enumerate(RHS):
                        if RHS[i][2:] == [n, max_mc]:
                            total_max_index.append(i)
                        if RHS[i][2:] == [n, min_mc]:
                            total_min_index.append(i)
                        if len(total_max_index) > 0:
                            max_index = np.random.choice(total_max_index)
                            RHS[max_index][3] = min_mc
                        if len(total_min_index) > 0:
                            min_index = np.random.choice(total_min_index)
                            RHS[min_index][3] = max_mc
    
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
        return self.population_dict