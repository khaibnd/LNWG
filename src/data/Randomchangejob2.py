import pandas as pd
import numpy as np
import random
from copy import deepcopy



input_link = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/input.xlsx'
link = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/output2.xlsx'
file_output = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/new_output.xlsx'
OSSM = 0.01




def read_file(link):
    observation = pd.read_excel(link, sheet_name='best_solution')
    return observation
    

def excel_writer(observation, file, sheet_name):
    writer = pd.ExcelWriter(file)
    observation.to_excel(writer, sheet_name)
    writer.save()


def sequence_import(input_link):
    sequence = pd.read_excel(input_link, sheet_name='sequence')
    return sequence


def get_part_sequence(sequence, part, num_sequence):
    part_sequence = sequence[(sequence.part == part) & (sequence.num_sequence == num_sequence)]
    part_sequence = part_sequence.dropna(axis=1, how='any')
    part_sequence = part_sequence.values.flatten().tolist()[2:]
    return part_sequence


def observation_sequence_sort(sequence, initial_observation):
        final_observation = pd.DataFrame()
        for lot in initial_observation['num_lot'].unique():
            # Get current part of lot
            part = initial_observation[
                initial_observation['num_lot'] == lot]['part'].values[0]
            num_sequence = initial_observation[
                initial_observation['num_lot'] == lot]['num_sequence'].values[0]
    
            # Get part production sequence
            part_sequence = get_part_sequence(sequence, part, num_sequence)
            # Split sublot of lot
            sub_df_per_lot = initial_observation.loc[initial_observation['num_lot'] == lot]
            # Get sub_df_per_lot index
            new_index_sequence = deepcopy(sub_df_per_lot.index.values)
            # Sort sub_df_per_lot by part sequence
            sub_df_per_lot['operation'] = pd.Categorical(
                sub_df_per_lot['operation'], part_sequence)
            sub_df_per_lot = sub_df_per_lot.sort_values('operation')
            sub_df_per_lot.index = new_index_sequence
            # Build element of population
            final_observation = final_observation.append(sub_df_per_lot)
            # Sort Dataframe by accending index
        final_observation = final_observation.sort_index()
        return final_observation


def main(link, input_link, OSSM):
    observation = read_file(link)
    observation_size = len(observation)
    if 1 > np.random.uniform(0, 1):
        observation_size = len(observation)
        print(observation_size)
        df_OSSM_pick = random.sample(list(range(0,observation_size)), int(observation_size*OSSM))
        df_OSSM_pick = sorted(df_OSSM_pick)
        print(df_OSSM_pick)
        for row_idx in df_OSSM_pick:
            row = observation[observation.index==row_idx]
            print(row)
            print(type(row))
            new_row_idx = random.choice(range(observation.shape[0]))
            print(row_idx, new_row_idx)

            child = pd.DataFrame(columns=['num_job',
                                          'num_lot',
                                          'part',
                                          'operation',
                                          'machine',
                                          'start_time',
                                          'completion_time'])

            # Swap position of OSSM pick to new random position in parent
            if new_row_idx > row_idx:
                child = child.append(observation.loc[0:row_idx-1,:], ignore_index=True)
                child = child.append(observation.loc[row_idx +1:new_row_idx,:], ignore_index=True)
                child = child.append(row, ignore_index=True)
                child = child.append(observation.loc[new_row_idx+1:,:], ignore_index=True)
                print(child.loc[row_idx,:])
    
            elif new_row_idx < row_idx:
                child = child.append(observation.loc[0:new_row_idx-1,:], ignore_index=True)
                child = child.append(row, ignore_index=True)
                child = child.append(observation.loc[new_row_idx:row_idx-1,:], ignore_index=True)
                child = child.append(observation.loc[row_idx+1:,:], ignore_index=True)

            sequence = sequence_import(input_link)
            
            observation = observation_sequence_sort(sequence, child)

    excel_writer(observation, file_output, sheet_name='OSSM')
    
    return observation


if __name__ == '__main__':
    print('Start!!')
    main = main(link, input_link, OSSM)