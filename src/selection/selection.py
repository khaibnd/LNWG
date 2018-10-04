'''k-way seletion of GA'''
import random
import numpy as np
from src.fitness_calculation.fitness_calculation import FitnessCalculation


# pylint: disable=E0401, R0913
class ChrosKWaySelection():
    '''1. k individuals are randomly selected
       2. The one presenting the highest fitness is declared the winner
       3. A copy of this individual is added to the mating pool to form the next generation.
       4. the k individuals in the tournament are placed back in the current population and the process is repeated.
       5. This continues until the number of individuals added to the mating pool is equal to the population size.'''
    def __init__(self, population_dict):
        self.population_dict = population_dict
        


    def generate_df_selection(self):
        '''Selection the chromosones from population for k-way selection'''

        
        new_population = {}
        population_size = int(self.parameter[self.parameter.name == 'population_size']['value'])
        k_way_parameter = int(self.parameter[self.parameter.name == 'k_way']['value'])
        population_key_list = self.population_dict.keys()
        population_makespan_dict = {num_observation:
                                    FitnessCalculation.
                                    calculate_weighted_tardiness(self,
                                                                 self.population_dict.get(num_observation))
                                    for num_observation in population_key_list}
        for new_num_observation in range(population_size):
            k_way_selection_dict = random.sample(population_makespan_dict.items(),
                                                 k_way_parameter)
            # Convert tuple to dict
            k_way_selection_dict = dict((x, y) for x, y in k_way_selection_dict)
            # Select minimum item in k_way_selection_dict
            min_k_way_selection_key = min(k_way_selection_dict,
                                          key=lambda key: k_way_selection_dict[key])
            # Append item to new population by index accending sequence
            new_population[new_num_observation] = self.population_dict[min_k_way_selection_key]
        return new_population


if __name__ == '__main__':
    a = ChrosKWaySelection()
    b = a.generate_df_selection()
    print(b)
