'''Making crossover inside population sample'''
import numpy as np


class ChrosCrossover():
    '''After mating pool has been generated using selection operator,
       1. the individuals are randomly paired to form parents for the next generator
       2. For each pair,  the algorithms selects one of the available crossover operators
          and applies it with a certain probability to create two child individuals by exchanging
          information contained in the parents'''
    def __init__(self, population_dict, parameter, sequence, machine):
        self.population_dict = population_dict
        self.parameter = parameter
        self.sequence = sequence
        self.machine = machine
        self.sequence_type = ['seq_1', 'seq_2', 'seq_3']


    def fast_copy(self, copy_value):
        '''Method fast copy to increase copy speed, instead of deepcopy'''
        output = copy_value.copy()
        for key, value in output.items():
            output[key] = ChrosCrossover.fast_copy(value) if isinstance(value, dict) else value
        return output


    def chros_crossover(self):
        '''Main function'''
        population_size = int(self.parameter[self.parameter.name == 'population_size']['value'])
        OMAC_rate = self.parameter[self.parameter.name == 'OMAC_rate']['value'].values[0]
        LSC_rate = self.parameter[self.parameter.name == 'LSC_rate']['value'].values[0]
        PSC_rate = self.parameter[self.parameter.name == 'PSC_rate']['value'].values[0]
        # pylint: disable=C0103
        # Generate random sequence to select the parent chromosome
        random_sequence = list(np.random.permutation(population_size))
        # Create child:
        
        for m in range(population_size//2):
            parent_index_1 = random_sequence[m]
            parent_index_2 = random_sequence[m + population_size//2]
            
            parent_1 = self.population_dict[parent_index_1]
            parent_2 = self.population_dict[parent_index_2]
            # Operation to machine Assignment Crossover
            if OMAC_rate > np.random.uniform(0, 1):

                OMAC_pick_parent_1 = parent_1.sample(int(population_size * OMAC_rate),
                                                     replace=False)
                OMAC_pick_parent_2 = parent_2.sample(int(population_size * OMAC_rate),
                                                     replace=False)

                for index, row in parent_1.iterrows():
                    for sample_index, sample_row in OMAC_pick_parent_2.iterrows():
                        if (row['num_lot'] == sample_row['num_lot']) &\
                        (row['operation'] == sample_row['operation']) &\
                        (row['num_sequence'] == sample_row['num_sequence']):
                            self.population_dict[parent_index_1].loc[index, 'machine'] = sample_row['machine']

                for index, row in parent_2.iterrows():
                    for sample_index, sample_row in OMAC_pick_parent_1.iterrows():
                        if (row['num_lot'] == sample_row['num_lot']) &\
                        (row['operation'] == sample_row['operation']) &\
                        (row['num_sequence'] == sample_row['num_sequence']):
                            self.population_dict[parent_index_2].loc[index, 'machine'] = sample_row['machine']  

            # Lot sequence Crossover
            if LSC_rate > np.random.uniform(0, 1):
                LSC_pick_parent_1 = parent_1.sample(n=1)
                operation_parent_1 = LSC_pick_parent_1['operation'].values[0]
                sequence_parent_1 = LSC_pick_parent_1['num_sequence'].values[0]
                LSC_pick_parent_2 = parent_2.sample(n=1)
                operation_parent_2 = LSC_pick_parent_2['operation'].values[0]
                sequence_parent_2 = LSC_pick_parent_2['num_sequence'].values[0]
                
                parent_1_pick_dict = parent_1.loc[(parent_1['operation'] != operation_parent_1) &
                                                  (parent_1['num_sequence'] != sequence_parent_1)]
                parent_1_pick_dict.reset_index(drop=True)
                parent_2_pick_dict = parent_2.loc[(parent_2['operation'] != operation_parent_2) &
                                                  (parent_2['num_sequence'] != sequence_parent_2)]
                parent_2_pick_dict.reset_index(drop=True)

                # Child 1 Creation
                for index, row in parent_1.iterrows():
                    if (row['operation'] != operation_parent_1) and (row['num_sequence'] != sequence_parent_1):
                        self.population_dict[parent_index_1].loc[index] = parent_1_pick_dict.iloc[0]
                        parent_1_pick_dict.drop(parent_1_pick_dict.index[0], inplace=True)
                        parent_1_pick_dict.reset_index(drop=True)
                # Child 2 Creation
                for index, row in parent_2.iterrows():
                    if (row['operation'] != operation_parent_2) and (row['num_sequence'] != sequence_parent_2):
                        self.population_dict[parent_index_2].loc[index] = parent_2_pick_dict.iloc[0]
                        parent_2_pick_dict.drop(parent_2_pick_dict.index[0], inplace=True)
                        parent_2_pick_dict.reset_index(drop=True)
  
            if PSC_rate > np.random.uniform(0, 1):

                PSC_pick_parent_1 = parent_1.sample(n=1)
                part_parent_1 = PSC_pick_parent_1['part'].values[0]
                sequence_parent_1 = PSC_pick_parent_1['num_sequence'].values[0]
                PSC_pick_parent_2 = parent_2.sample(n=1)
                part_parent_2 = PSC_pick_parent_2['part'].values[0]
                sequence_parent_2 = PSC_pick_parent_2['num_sequence'].values[0]
                
                parent_1_pick_dict = parent_1.loc[(parent_1['part'] != part_parent_1) &
                                                  (parent_1['num_sequence'] != sequence_parent_1)]
                parent_1_pick_dict.reset_index(drop=True)
                parent_2_pick_dict = parent_2.loc[(parent_2['part'] != part_parent_2) &
                                                  (parent_2['num_sequence'] != sequence_parent_2)]
                parent_2_pick_dict.reset_index(drop=True)

                # Child 1 Creation
                for index, row in parent_1.iterrows():
                    if (row['part'] != part_parent_1) and (row['num_sequence'] != sequence_parent_1):
                        self.population_dict[parent_index_1].loc[index] = parent_1_pick_dict.iloc[0]
                        parent_1_pick_dict.drop(parent_1_pick_dict.index[0], inplace=True)
                        parent_1_pick_dict.reset_index(drop=True)
                # Child 2 Creation
                for index, row in parent_2.iterrows():
                    if (row['part'] != part_parent_2) and (row['num_sequence'] != sequence_parent_2):
                        self.population_dict[parent_index_2].loc[index] = parent_2_pick_dict.iloc[0]
                        parent_2_pick_dict.drop(parent_2_pick_dict.index[0], inplace=True)
                        parent_2_pick_dict.reset_index(drop=True)
    
        return self.population_dict
