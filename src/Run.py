'''Main run file'''
import io
import os
import pstats
import cProfile
import pandas as pd
from copy import deepcopy
from timeit import default_timer as timer
#sys.path.extend(['/Users/khaibnd/eclipse-workspace/LNWG4'])

from src.data.data import DataInput
from src.data.data import DataOutput
from src.data.data import ExcelFile
from src.initial_solution.initial_solution import InitialSolution
from src.selection.selection import ChrosKWaySelection
from src.crossover.crossover import ChrosCrossover
from src.mutation.mutation import ChrosMutation
from src.fitness_calculation.fitness_calculation import FitnessCalculation
from src.gantt.plotly_gantt import PlotlyGantt


if os.path.isdir('/Users/khaibnd/eclipse-workspace/LNWG4/src/data/'):
    FOLDER = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/'
else:
    FOLDER = r'C:\\Users\\khai.bui\\Desktop\\Operations\\5. Eclipse\\LNWG\\src\\data\\'
    
INPUT_FILE_LINK = FOLDER + 'input.xlsx'
OUTPUT_FILE_LINK = FOLDER + 'output.xlsx'
INITIAL_OUTPUT = FOLDER + 'Initial.xlsx'
SELECTION_OUTPUT = FOLDER + 'Selection.xlsx'
CROSSOVER_OUTPUT = FOLDER + 'Crossover.xlsx'
MUTATION_OUTPUT = FOLDER + 'Mutation.xlsx'
ITERATION_OUTPUT = FOLDER + 'Iteration.xlsx'
CPROFILE_LOG = FOLDER + 'cprofile_log.txt'

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
        self.machine_lotsize,\
        self.processing_time,\
        self.wip,\
        self.finished_goods = DataInput().data_reader(INPUT_FILE_LINK)
    def delete_old_files(self):
        ExcelFile.file_remove(self, OUTPUT_FILE_LINK)
        ExcelFile.file_remove(self, INITIAL_OUTPUT)
        ExcelFile.file_remove(self, SELECTION_OUTPUT)
        ExcelFile.file_remove(self, CROSSOVER_OUTPUT)
        ExcelFile.file_remove(self, ITERATION_OUTPUT)
        ExcelFile.file_remove(self, CPROFILE_LOG)

    def main_run(self):
        '''Main function to get output'''
        self.population_tardiness_dict = {}
        best_solution = None
        global_best_tardiness = 999999999999999999
        num_iteration = int(self.parameter[self.parameter.name == 'num_iteration']['value'])
        # Gemerate initial population
        LoadInitial = InitialSolution(self.parameter,
                                      self.demand,
                                      self.wip,
                                      self.criteria,
                                      self.machine,
                                      self.sequence)
        self.population_dict = LoadInitial.generate_initial_population()
        DataOutput.operation_output_writer(self, INITIAL_OUTPUT)

        initial_finished_time = timer() - program_start
        print('Initial Solution Generated')

        iteration_record_columns = ['worst_tardiness', 'best_tardiness', 'global_best_tardiness', 'iteration_run_time']
        iteration_record = pd.DataFrame(columns=iteration_record_columns)

        for iteration in range(num_iteration):
            iteration_start = timer()
            print('Iteration#:',iteration)

            # k-way selection
            self.population_dict = ChrosKWaySelection.generate_df_selection(self, iteration)
            # DataOutput.operation_output_writer(self, SELECTION_OUTPUT)
            print('k-way selection Generated')

            # crossover
            self.population_dict = ChrosCrossover.chros_crossover(self)
            print('Crossover Generated')

            #mutation
            self.population_dict = ChrosMutation.chros_mutation(self)
            # DataOutput.operation_output_writer(self, MUTATION_OUTPUT)
            print('Mutation Generated')

            # fitness calculation to find the optimal solution
            local_tardiness_list = []

            for num_observation, observation in iter(self.population_dict.items()):
                
                local_tardiness = FitnessCalculation.calculate_weighted_tardiness(self, observation)
                self.population_tardiness_dict[num_observation] = local_tardiness
                print('o_%s: %s' %(num_observation,local_tardiness))
                local_tardiness_list.append(local_tardiness)
                if local_tardiness < global_best_tardiness:
                    
                    global_best_tardiness = deepcopy(local_tardiness)
                    best_solution = deepcopy(observation)
                    print('global best tardiness: ', global_best_tardiness)
                else:
                    pass
            local_worst_tardiness = max(local_tardiness_list)
            local_best_tardiness = min(local_tardiness_list)
            iteration_finished = timer()
            iteration_run_time = iteration_finished - iteration_start
            
            new_row_iteration_record = pd.DataFrame([[local_worst_tardiness,
                                                      local_best_tardiness,
                                                      global_best_tardiness,
                                                      iteration_run_time]],
                                                      columns=iteration_record_columns)
            iteration_record = iteration_record.append(new_row_iteration_record, ignore_index=True)
            
            DataOutput.iteration_record_writer(self, ITERATION_OUTPUT, iteration_record, best_solution)

        # Adding timeline to best solution
        genetic_finished_time = timer() - program_start
        
        DataOutput.data_writer(self, OUTPUT_FILE_LINK,
                               global_best_tardiness,
                               best_solution,
                               initial_finished_time,
                               genetic_finished_time)
        # Build plot.ly Gantt Chart
        # PlotlyGantt.plotly_gantt(self, best_solution)

        return global_best_tardiness, best_solution

if __name__ == '__main__':
    print("Start!!!")
    profile_program = cProfile.Profile()
    profile_program.enable()
    main = main()
    main.delete_old_files()
    main.main_run()
    profile_program.disable()
    written_data = io.StringIO()
    line_written_data = pstats.Stats(profile_program,
                                     stream=written_data).sort_stats('tottime')
    line_written_data.print_stats()
    with open(CPROFILE_LOG, 'w+') as output_log_file:
        output_log_file.write(written_data.getvalue())
    print('Finished!!')
