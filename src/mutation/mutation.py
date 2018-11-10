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
    '''
    1. Each child chromosome with small probabilities 
    2. Each single chromosome at the gene level to alter information contained in the gene.
    '''

    def __itit__(self, population_dict, parameter, sequence, machine):
        self.population_dict = population_dict
        self.parameter = parameter
        self.sequence = sequence
        self.machine = machine


    def machine_list(self):
        operation_list = self.machine.columns.tolist()[1:]
        machine_per_operation = {operation: sta(self.machine[self.machine['criteria']\
                                            == 'num_machine'][operation].values[0])
                                                             for operation in operation_list}

        machine_list = set([i[0] for i in machine_per_operation.values()])
        machine_list = sorted(machine_list)
        return machine_list


    def machine_assign_time(self):
        '''Randomly selection machine for operation from list of num_machine'''
        machine_list = ChrosMutation.machine_list(self)
        return {operation: 0 for operation in machine_list}


    def chros_mutation(self):
        '''Make mutation for all the population'''
        ROAM = self.parameter[self.parameter.name == 'ROAM']['value'].values[0]
        OSSM = self.parameter[self.parameter.name == 'OSSM']['value'].values[0]
        OSSM_size = self.parameter[self.parameter.name == 'OSSM_size']['value'].values[0]
        OSSM_size = int(OSSM_size)
        IJMM = self.parameter[self.parameter.name == 'IJMM']['value'].values[0]
        IJMM_size = self.parameter[self.parameter.name == 'IJMM_size']['value'].values[0]
        IJMM_size = int(IJMM_size)
        ITSM = self.parameter[self.parameter.name == 'ITSM']['value'].values[0]


        for chros_index, chros in iter(self.population_dict.items()):

            chros_size = len(chros)

            # Random Operation Assignment Mutation (ROAM)
            if ROAM > np.random.uniform(0, 1):
                '''
                Reason: Balance assignment frequency for all machine in each operation.

                1. Check  assignment frequency of machine type from start to current gene
                2. Get the minimum frequency machine#
                3. Change current machine assignment to the minimum one.
                '''
                df_ROAM_pick = chros.sample(int(chros_size * ROAM),
                                                 replace=False)
                df_ROAM_pick = df_ROAM_pick.sort_index()
                for gene_index, gene in df_ROAM_pick.iterrows():
                    gene_num_sequence = gene['num_sequence']
                    gene_operation = gene['operation']
                    chros_same_operation = chros.loc[(chros.operation == gene_operation)
                                                               & (chros.num_sequence == gene_num_sequence)]
                    chros_same_operation_freq = chros_same_operation['machine'].value_counts()
                    if chros_same_operation_freq.sum() > 0:
                        assigned_machine = chros_same_operation_freq.idxmin()
                        chros.loc[gene_index, 'machine'] = assigned_machine


            # Operations Sequence Shift Mutation (OSSM)
            
            if OSSM > np.random.uniform(0, 1):
                '''
                Reason: Change position numbers of gene position to new one
                        w/o affecting production sequence.

                1. Get randomly list of gene index in chromosome with predefine size
                2. Ascending Sort the list
                3. For each element of list, define the gene. Get randomly new gene position index
                and swap together
                '''
                chros_index_list = list(range(0, chros_size))
                df_OSSM_pick = random.sample(chros_index_list, OSSM_size)
                df_OSSM_pick = sorted(df_OSSM_pick)

                for gene_index in df_OSSM_pick:
                    gene = chros[chros.index == gene_index]
                    new_gene_index = random.choice(range(chros.shape[0]))

                    new_chros = pd.DataFrame(columns=['num_job',
                                                  'num_lot',
                                                  'part',
                                                  'operation',
                                                  'machine'])
        
                    # Swap position of OSSM pick to new random position in parent
                    if new_gene_index > gene_index:
                        new_chros = new_chros.append(chros.loc[0:gene_index - 1, :], ignore_index=True)
                        new_chros = new_chros.append(chros.loc[gene_index + 1:new_gene_index, :], ignore_index=True)
                        new_chros = new_chros.append(gene, ignore_index=True)
                        new_chros = new_chros.append(chros.loc[new_gene_index + 1:, :], ignore_index=True)
            
                    elif new_gene_index < gene_index:
                        new_chros = new_chros.append(chros.loc[0:new_gene_index - 1, :], ignore_index=True)
                        new_chros = new_chros.append(gene, ignore_index=True)
                        new_chros = new_chros.append(chros.loc[new_gene_index:gene_index - 1, :], ignore_index=True)
                        new_chros = new_chros.append(chros.loc[gene_index + 1:, :], ignore_index=True)
                    else:
                        new_chros = chros
                    
                    chros = new_chros
                chros = InitialSolution.chros_sequence_sort(self, chros)


            # Inteligent Job Machine Mutation (IJMM)            
            if  IJMM > np.random.uniform(0, 1):
                '''
                Reason: Changing job priority base on demand priority on the gene order.

                1. Get randomly number of gene in chromosome w/o replacement with predefine sizes
                2. For each element in above DF:
                    - Get previous and precedent operation index
                    - Get DF inside previous and precedent operation index with same machine
                    - Check each gene in above DF a that can swap together.
                    - Re-define gene index base on the demand priority
                3. Update new chromosome index
                '''
                IJMM_pick_df = chros.sample(IJMM_size, replace=False)
                IJMM_pick_df = IJMM_pick_df.sort_index()
                for gene_index, gene in IJMM_pick_df.iterrows():
                    gene_part = gene['part']
                    gene_num_lot = gene['num_lot']
                    gene_operation = gene['operation']
                    gene_machine = gene['machine']
                    gene_num_sequence = gene['num_sequence']
                    prev_gene_operation = FitnessCalculation.prev_part_sequence_operation(self, gene_part, gene_operation, gene_num_sequence)
                    post_gene_operation = FitnessCalculation.post_part_sequence_operation(self, gene_part, gene_operation, gene_num_sequence)
        
                    try:
                        gene_min_index = chros.index[(chros.num_lot == gene_num_lot)
                                                        & (chros.operation == prev_gene_operation)].tolist()[0]
                    except:
                        gene_min_index = 0
                    try:
                        gene_max_index = chros.index[(chros.num_lot == gene_num_lot)
                                                        & (chros.operation == post_gene_operation)].tolist()[0]
                    except:
                        gene_max_index = len(chros)
        
                    check_machine_df = chros.loc[(chros.machine == gene_machine)
                                                      & (chros.num_sequence == gene_num_sequence)
                                                      & (chros.index > gene_min_index)
                                                      & (chros.index < gene_max_index)]
                    
                    review_df = chros[gene_min_index:gene_max_index]
                    df = pd.DataFrame()
                    for _, check_gene in check_machine_df.iterrows():
                        check_gene_part = check_gene['part']
                        check_gene_num_sequence = check_gene['num_sequence']
                        check_gene_num_lot = check_gene['num_lot']

                        check_gene_operation = check_gene['operation']
        
                        check_prev_gene_operation = FitnessCalculation.prev_part_sequence_operation(self, check_gene_part, check_gene_operation, check_gene_num_sequence)
                        try:
                            check_prev_gene_operation_index = review_df.index[(review_df.num_lot == check_gene_num_lot)
                                                        & (review_df.operation == check_prev_gene_operation)].tolist()[0]
                        except:
                            check_prev_gene_operation_index = None

                        check_post_gene_operation = FitnessCalculation.post_part_sequence_operation(self, check_gene_part, check_gene_operation, check_gene_num_sequence)

                        try:
                            check_post_gene_operation_index = review_df.index[(review_df.num_lot == check_gene_num_lot)
                                                        & (review_df.operation == check_post_gene_operation)].tolist()[0]
                        except:
                            check_post_gene_operation_index = None

                        if ((check_prev_gene_operation_index == None) and
                            (check_post_gene_operation_index == None)):
                            df = df.append(check_gene)

                    if (len(df) >= 2):
                        df_index = df.index.tolist()
                        df['num_job'] = df['num_job'].astype(int)
                        df = df.sort_values(['num_job'])
                        df.index = df_index
                    chros.update(df)


            # Intelligent Task Sort Mutation (ITSM)
            if ITSM > np.random.uniform(0, 1):
                '''
                Reason: Redefine lot order to the gene as demand priority.

                1. Get parts list of chromosome
                2. For each part get completion index of each lot to the list
                3. Sorting list as demand priority order using Quick Sort Algorithms
                4. For each element in new sorted list:
                    - Get all elements of removed lot (old) and incoming lot (new) to DF
                    - Balance the length of 2 DF
                    - Moving the new values to the old index as same operation and append to new DF
                    - Keep left over if new length less than old length (keep original index)
                5. Adding leftover to new chromosome index
                6. Ascending sorting and redefine new natural index
                7. Updating the chromosome as new
                '''
                part_list = sorted(chros['part'].unique())
                leftover_df = pd.DataFrame()
                temp_df = pd.DataFrame()
                for part in part_list:
                    group_part_job = chros.loc[chros.part == part][:]
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
                                GREATER = [element for element in PACKING_INDEX_LIST[1:] if int(group_num_sequence.at[element, 'num_job']) >= int(group_num_sequence.at[PIVOT, 'num_job'])]
                                LESSER = [element for element in PACKING_INDEX_LIST[1:] if int(group_num_sequence.at[element, 'num_job']) < int(group_num_sequence.at[PIVOT, 'num_job'])]
                                return quick_sort(LESSER, group_num_sequence) + [PIVOT] + quick_sort(GREATER, group_num_sequence)
                            
                        new_packing_index_list = quick_sort(packing_index_list, group_num_sequence)
                        
                        for index in range(len(packing_index_list)):
                            new_pack_index = new_packing_index_list[index]
                            old_pack_index = packing_index_list[index]

                            old_lot = group_num_sequence.at[old_pack_index, 'num_lot']
                            old_df = deepcopy(group_num_sequence[group_num_sequence['num_lot'] == old_lot].sort_index())
                            
                            new_lot = group_num_sequence.at[new_pack_index, 'num_lot']
                            new_df = deepcopy(group_num_sequence[group_num_sequence['num_lot'] == new_lot].sort_index())

                            get_length = min(len(old_df), len(new_df))

                            if len(old_df) >= len(new_df):
                                old_df = old_df[-get_length:]

                            else:
                                leftover_df = leftover_df.append(new_df[:-get_length])
                                new_df = new_df[-get_length:]

                            new_df.index = old_df.index.tolist() 
                            temp_df = temp_df.append(new_df)
                                
                temp_full = temp_df.append(leftover_df)
                temp_full = temp_full.sort_index()
                temp_full = temp_full.reset_index(drop=True)
                chros.update(temp_full)
            #print('mutation len', len(chros))
            self.population_dict[chros_index] = chros
        
        return self.population_dict
