'''Main run file'''
import io
import pstats
import cProfile
import sys
import pandas as pd
from copy import deepcopy
from timeit import default_timer as timer
#sys.path.extend(['/Users/khaibnd/eclipse-workspace/LNWG4'])

from src.data.data import DataInput
from src.data.data import DataOutput
from src.initial_solution.initial_solution import InitialSolution
from src.selection.selection import ChrosKWaySelection
from src.crossover.crossover import ChrosCrossover
from src.mutation.mutation import ChrosMutation
from src.fitness_calculation.fitness_calculation import FitnessCalculation
from src.gantt.plotly_gantt import PlotlyGantt


INPUT_FILE_LINK = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/input.xlsx'
OUTPUT_FILE_LINK = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/output.xlsx'
INITIAL_OUTPUT = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/InitialSolution.xlsx'
SELECTION_OUTPUT = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/Selection.xlsx'
CROSSOVER_OUTPUT = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/Crossover.xlsx'
MUTATION_OUTPUT = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/Mutation.xlsx'
ITERATION_OUTPUT = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/Iteration.xlsx'
CPROFILE_LOG = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/cprofile_log.txt'
# pylint: disable=E0401
program_start = timer()
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
        self.population_tardiness_dict = {}
        best_solution = None
        best_tardiness = 999999999999999
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
        initial_finished_time = timer() - program_start

        print('Initial Solution Generated')
        iteration_record_columns = ['worst_tardiness', 'best_tardiness', 'global_best_tardiness', 'iteration_run_time']
        iteration_record = pd.DataFrame(columns=iteration_record_columns)
        for iteration in range(num_iteration):
            iteration_start = timer()
            print('iteration#:',iteration)
            # k-way selection
            
            self.population_dict = ChrosKWaySelection.generate_df_selection(self, iteration)
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
            local_tardiness_list = []
            
            for num_observation, observation in iter(self.population_dict.items()):
                local_tardiness = FitnessCalculation.calculate_weighted_tardiness(self, observation)
                self.population_tardiness_dict[num_observation] = local_tardiness
                print(local_tardiness)
                local_tardiness_list.append(local_tardiness)
                if local_tardiness < best_tardiness:
                    best_tardiness = deepcopy(local_tardiness)
                    best_solution = deepcopy(observation)
                    print('best local tardiness: ', best_tardiness)
            worst_tardiness = max(local_tardiness_list)
            best_tardiness = min(local_tardiness_list)
            global_best_tardiness = best_tardiness
            iteration_finished = timer()
            iteration_run_time = iteration_finished - iteration_start
            
            new_row_iteration_record = pd.DataFrame([[worst_tardiness,
                                                      best_tardiness,
                                                      global_best_tardiness,
                                                      iteration_run_time]],
                                                      columns=iteration_record_columns)
            iteration_record = iteration_record.append(new_row_iteration_record, ignore_index=True)
            
            DataOutput.iteration_record_writer(self, ITERATION_OUTPUT, iteration_record, best_solution)
        # Adding timeline to best solution
        genetic_finished_time = timer() - program_start
        
        DataOutput.data_writer(self, OUTPUT_FILE_LINK,
                               best_tardiness,
                               best_solution,
                               initial_finished_time,
                               genetic_finished_time)
        # Build plot.ly Gantt Chart
        # PlotlyGantt.plotly_gantt(self, best_solution)

        return best_tardiness, best_solution

if __name__ == '__main__':
    profile_program = cProfile.Profile()
    profile_program.enable()
    main = main()
    main_run = main.main_run()
    profile_program.disable()
    written_data = io.StringIO()
    line_written_data = pstats.Stats(profile_program,
                                     stream=written_data).sort_stats('tottime')
    line_written_data.print_stats()
    with open(CPROFILE_LOG, 'w+') as output_log_file:
        output_log_file.write(written_data.getvalue())
    
