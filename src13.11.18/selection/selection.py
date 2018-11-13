'''k-way seletion of GA'''
import random
from src.fitness_calculation.fitness_calculation import FitnessCalculation


# pylint: disable=E0401, R0913
class ChrosKWaySelection():
    '''1. k individuals are randomly selected
       2. The one presenting the highest fitness is declared the winner
       3. A copy of this individual is added to the mating pool to form the next generation.
       4. the k individuals in the tournament are placed back in the current population and the process is repeated.
       5. This continues until the number of individuals added to the mating pool is equal to the population size.'''

    def __init__(self, population_dict, population_tardiness_dict, iteration):
        self.population_dict = population_dict
        self.population_tardiness_dict = population_tardiness_dict
        self.iteration = iteration     

    def generate_df_selection(self, iteration, old_population, old_population_tardiness):
        '''Selection the chromosones from population for k-way selection'''
        
        new_population = {}
        population_size = int(self.parameter[self.parameter.name == 'population_size']['value'])
        k_way_parameter = int(self.parameter[self.parameter.name == 'k_way']['value'])

        if iteration == 0:
            population_key_list = self.population_dict.keys()
            self.population_tardiness_dict = {num_observation:
                                        FitnessCalculation.
                                        calculate_weighted_tardiness(self,
                                                                     self.population_dict.get(num_observation))
                                        for num_observation in population_key_list}

        # parent = {**self.population_dict, **old_population}
        # parent_tardiness = {**self.population_tardiness_dict, **old_population_tardiness}

        # for python <=3.4, PEP448
        def merge_2_dicts(x, y):
            z = x.copy()
            z.update(y)
            return z

        parent = merge_2_dicts(self.population_dict, old_population)
        parent_tardiness = merge_2_dicts(self.population_tardiness_dict, old_population_tardiness)

        for new_num_observation in range(population_size):
            k_way_selection_dict = random.sample(parent_tardiness.items(),
                                                 k_way_parameter)
            # Convert tuple to dict
            k_way_selection_dict = dict((x, y) for x, y in k_way_selection_dict)
            # Select minimum item in k_way_selection_dict
            k_way_selection_key = max(k_way_selection_dict,
                                          key=lambda key: k_way_selection_dict[key])
            # Append item to new population by index accending sequence
            new_population[new_num_observation] = parent[k_way_selection_key]
            old_population[new_num_observation + population_size] = parent[k_way_selection_key]
            old_population_tardiness[new_num_observation] = k_way_selection_key
        return new_population, old_population, old_population_tardiness


if __name__ == '__main__':
    a = ChrosKWaySelection()
    b = a.generate_df_selection()
    print(b)
