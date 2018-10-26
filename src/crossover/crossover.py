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
            parent_1 = self.population_dict[random_sequence[m]]
            parent_1['num_job'] = parent_1['num_job'].astype(int)
            parent_2 = self.population_dict[random_sequence[m + population_size//2]]
            parent_2['num_job'] = parent_2['num_job'].astype(int)
            child_1 = ChrosCrossover.fast_copy(self, parent_1)
            child_2 = ChrosCrossover.fast_copy(self, parent_2)

            # Operation to machine Assignment Crossover
            if OMAC_rate > np.random.uniform(0, 1):

                OMAC_pick_parent_1 = parent_1.sample(int(population_size * OMAC_rate),
                                                     replace=False)
                OMAC_pick_parent_2 = parent_2.sample(int(population_size * OMAC_rate),
                                                     replace=False)

                for child_index, child_row in child_1.iterrows():
                    for sample_index, sample_row in OMAC_pick_parent_2.iterrows():
                        if (child_row['num_lot'] == sample_row['num_lot'] and
                            child_row['operation'] == sample_row['operation'] and
                            child_row['machine'] != sample_row['machine']):
                            child_row['machine'] = sample_row['machine']

                for child_index, child_row in child_2.iterrows():
                    for sample_index, sample_row in OMAC_pick_parent_1.iterrows():
                        if (child_row['num_lot'] == sample_row['num_lot'] and
                            child_row['operation'] == sample_row['operation'] and
                            child_row['machine'] != sample_row['machine']):
                            child_row['machine'] = sample_row['machine']
                parent_1 = child_1
                parent_2 = child_2

            # Lot sequence Crossover
            if np.random.uniform(0, 1) < LSC_rate:
                '''Operation sequence crossover'''
                LSC_pick_parent_1 = parent_1.sample(n=1)
                LSC_pick_parent_1 = LSC_pick_parent_1['operation'].values[0]
                LSC_pick_parent_2 = parent_2.sample(n=1)
                LSC_pick_parent_2 = LSC_pick_parent_2['operation'].values[0]

                parent_1_pick_dict = parent_1.loc[parent_1['operation'] != LSC_pick_parent_1]
                parent_1_pick_dict.reset_index()
                parent_2_pick_dict = parent_1.loc[parent_1['operation'] != LSC_pick_parent_2]
                parent_2_pick_dict.reset_index()

                # Child 1 Creation
                for child_index, child_row in child_1.iterrows():
                    if child_row['operation'] != LSC_pick_parent_1:
                        child_row = parent_1_pick_dict.iloc[[0]]
                        parent_1_pick_dict.drop(parent_1_pick_dict.index[0])
                        parent_1_pick_dict.reset_index()
                # Child 2 Creation
                for child_index, child_row in child_2.iterrows():
                    if child_row['operation'] != LSC_pick_parent_2:
                        child_row = parent_2_pick_dict.iloc[[0]]
                        parent_2_pick_dict.drop(parent_2_pick_dict.index[0])
                        parent_2_pick_dict.reset_index()
    
                parent_1 = child_1
                parent_2 = child_2
                
            if np.random.uniform(0, 1) < PSC_rate:
                '''part crossover'''
                PSC_pick_parent_1 = parent_1.sample(n=1)
                PSC_pick_parent_1 = PSC_pick_parent_1['part'].values[0]
                PSC_pick_parent_2 = parent_2.sample(n=1)
                PSC_pick_parent_2 = PSC_pick_parent_2['part'].values[0]
            
                parent_1_pick_dict = parent_1.loc[parent_1['part'] != PSC_pick_parent_1]
                parent_1_pick_dict.reset_index()
                parent_2_pick_dict = parent_1.loc[parent_1['part'] != PSC_pick_parent_2]
                parent_2_pick_dict.reset_index()

                # Child 1 Creation
                for child_index, child_row in child_1.iterrows():
                    if child_row['part'] != PSC_pick_parent_1:
                        child_row = parent_1_pick_dict.iloc[[0]]
                        parent_1_pick_dict.drop(parent_1_pick_dict.index[0])
                        parent_1_pick_dict.reset_index()
                # Child 2 Creation
                for child_index, child_row in child_2.iterrows():
                    if child_row['part'] != PSC_pick_parent_2:
                        child_row = parent_2_pick_dict.iloc[[0]]
                        parent_2_pick_dict.drop(parent_2_pick_dict.index[0])
                        parent_2_pick_dict.reset_index()
    
                parent_1 = child_1
                parent_2 = child_2

        return self.population_dict
