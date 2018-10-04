'''Main run file'''
from src.data.data import DataInput
from src.data.data import DataOutput
from src.initial_solution.initial_solution import InitialSolution
from src.selection.selection import ChrosKWaySelection
from src.crossover.crossover import ChrosCrossover
from src.mutation.mutation import ChrosMutation
from src.fitness_calculation.fitness_calculation import FitnessCalculation
from src.gantt.plotly import PlotlyGantt
from copy import deepcopy
import pandas as pd

INPUT_FILE_LINK = r'/Users/khaibnd/eclipse-workspace/LNWG3/src/data/input.xlsx'
OUTPUT_FILE_LINK = r'/Users/khaibnd/eclipse-workspace/LNWG3/src/data/output.xlsx'
INITIAL_OUTPUT = r'/Users/khaibnd/eclipse-workspace/LNWG3/src/data/InitialSolution.xlsx'
SELECTION_OUTPUT = r'/Users/khaibnd/eclipse-workspace/LNWG3/src/data/Selection.xlsx'
CROSSOVER_OUTPUT = r'/Users/khaibnd/eclipse-workspace/LNWG3/src/data/Crossover.xlsx'
MUTATION_OUTPUT = r'/Users/khaibnd/eclipse-workspace/LNWG3/src/data/Mutation.xlsx'
# pylint: disable=E0401
class main():
    def __init__(self):
        self.parameter,\
        self.demand,\
        self.criteria,\
        self.sequence,\
        self.machine,\
        self.machine_criteria,\
        self.processing_time,\
        self.wip,\
        self.finished_goods = DataInput().data_reader(INPUT_FILE_LINK)


    def main_run(self):
        '''Main function to get output'''
        best_solution = None
        best_makespan = 999999999999999
        num_iteration = int(self.parameter[self.parameter.name == 'num_iteration']['value'])
        # Gemerate initial population
        LoadInitial = InitialSolution(self.parameter,
                                      self.demand,
                                      self.wip,
                                      self.criteria,
                                      self.machine,
                                      self.sequence)
        self.population_dict = LoadInitial.generate_initial_population()
        
        d = deepcopy(self.population_dict)
        writer = pd.ExcelWriter(INITIAL_OUTPUT)
        for i in range(20):
            e = d[i]
            e.to_excel(writer, sheet_name='s_%s' %i)
            writer.save()
        writer.save()
        '''
        for num_observation, observation in self.population_dict.items():
            a = (FitnessCalculation.calculate_weighted_tardiness(self, observation))
            print('Observation: %s, weight tardiness: %s' %( num_observation, a))
        '''
        print('Initial Solution Generated')
        
        for iteration in range(num_iteration):
            print('iteration#:',iteration)
            # k-way selection

            self.population_dict = ChrosKWaySelection.generate_df_selection(self)
            d = deepcopy(self.population_dict)
            writer = pd.ExcelWriter(SELECTION_OUTPUT)
            for i in range(20):
                e = d[i]
                e.to_excel(writer, sheet_name='s_%s' %i)
                writer.save()
            writer.save()
            print('k-way selection Generated')
            # crossover
            self.population_dict = ChrosCrossover.chros_crossover(self)
            print('Crossover Generated')
            #mutation
            self.population_dict = ChrosMutation.chros_mutation(self)
            writer = pd.ExcelWriter(MUTATION_OUTPUT)
            for i in range(20):
                e = d[i]
                e.to_excel(writer, sheet_name='s_%s' %i)
                writer.save()
            writer.save()
            print('Mutation Generated')

            # fitness calculation to find the optimal solution
            for _, observation in iter(self.population_dict.items()):
                local_makespan = FitnessCalculation.calculate_weighted_tardiness(self, observation)
                print(local_makespan)
                if local_makespan < best_makespan:
                    best_makespan = deepcopy(local_makespan)
                    best_solution = deepcopy(observation)
                    print('best local makespan: ', best_makespan)

        # Adding timeline to best solution
        best_solution = DataOutput.data_writer(self, OUTPUT_FILE_LINK, best_makespan, best_solution)
        # Build plot.ly Gantt Chart
        PlotlyGantt.plotly_gantt(self, best_solution)

        return best_makespan, best_solution

if __name__ == '__main__':
    main = main()
    main_run = main.main_run()
    print('main run: ', main_run)
