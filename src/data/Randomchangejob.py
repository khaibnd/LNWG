import pandas as pd
import numpy as np
from copy import deepcopy
from src.fitness_calculation.fitness_calculation import FitnessCalculation


input_link = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/input.xlsx'
link = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/output2.xlsx'
file_output = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/new_output.xlsx'
ROAM = 0.2




def read_file(link):
    observation = pd.read_excel(link, sheet_name='best_solution')
    return observation


def sequence_import(input_link):
    sequence = pd.read_excel(input_link, sheet_name='sequence')
    return sequence
    

def excel_writer(observation, file, sheet_name):
    writer = pd.ExcelWriter(file)
    observation.to_excel(writer, sheet_name)
    writer.save()

def part_sequence(sequence, part, num_sequence):
    part_sequence = sequence[(sequence.part == part) & (sequence.num_sequence == num_sequence)]
    part_sequence = part_sequence.dropna(axis=1, how='any')
    part_sequence = part_sequence.values.flatten().tolist()[2:]
    return part_sequence

def main(link, input_link, IOAM):
    observation = read_file(link)
    observation_size = len(observation)
    if 1 > np.random.uniform(0,1):
        df_ROAM_pick = observation.sample(int(observation_size*ROAM),
                                                replace=False)
        df_ROAM_pick = df_ROAM_pick.sort_index()
        for row_idx, row in df_ROAM_pick.iterrows():
            row_num_sequence = row['num_sequence']
            row_operation = row['operation']
            observation_same_operation = observation.loc[(observation.operation == row_operation)
                                                         & (observation.num_sequence == row_num_sequence)]
            print(observation_same_operation[['machine']])
            observation_same_operation_freq = observation_same_operation['machine'].value_counts()
            print('observation_same_operation_freq', observation_same_operation_freq)
            print(type(observation_same_operation_freq))
            if observation_same_operation_freq.sum() > 0:
                assigned_machine = observation_same_operation_freq.idxmin()
                print(assigned_machine)
                observation.loc[row_idx, 'machine'] = assigned_machine
    
    excel_writer(observation, file_output, sheet_name='ROAM')
    
    return observation


if __name__ == '__main__':
    print('Start!!')
    main = main(link, input_link, ROAM)