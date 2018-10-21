import pandas as pd
import numpy as np
from copy import deepcopy


link = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/output2.xlsx'
file_output = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/new_output.xlsx'
IOAM = 1
def read_file(link):
    observation = pd.read_excel(link, sheet_name='best_solution')
    return observation


def excel_writer(obervation, file, sheet_name):
    writer = pd.ExcelWriter(file)
    obervation.to_excel(writer, sheet_name)
    writer.save()

def main(link, IOAM):
    observation = read_file(link)

    if IOAM > np.random.uniform(0,1):
        part_list = sorted(observation['part'].unique())
        observation_list = observation.index.tolist()
        check1 = pd.DataFrame()
        check2 = pd.DataFrame()
        print('observation_list', observation_list)
        temp_full = pd.DataFrame()
        for part in part_list:
            group_part_job = observation.loc[observation.part == part]
            num_sequence_list = sorted(group_part_job['num_sequence'].unique())
            
            for num_sequence in num_sequence_list:
                group_num_sequence = group_part_job.loc[group_part_job.num_sequence == num_sequence]
                packing_index_list = group_num_sequence.index[group_num_sequence.operation == 'packing'].tolist()
                print(packing_index_list)
                def buble_sort(PACKING_INDEX_LIST, group_num_sequence, observation):
                    '''Using code from github.com/TheAlgorithms/Python/blob/master/sorts/quick_sort.py'''
                    length = len(PACKING_INDEX_LIST)
                    for i in range(length):
                        for j in range(0, length - i - 1):
                            temp_df = pd.DataFrame()
                            if group_num_sequence.at[PACKING_INDEX_LIST[j], 'num_job'] >= group_num_sequence.at[PACKING_INDEX_LIST[j + 1], 'num_job']:
                                print('old', PACKING_INDEX_LIST[j], PACKING_INDEX_LIST[j + 1])
                                PACKING_INDEX_LIST[j], PACKING_INDEX_LIST[j + 1] = PACKING_INDEX_LIST[j + 1], PACKING_INDEX_LIST[j]
                                print('old', PACKING_INDEX_LIST[j], PACKING_INDEX_LIST[j + 1])
                                old_lot = group_num_sequence.at[PACKING_INDEX_LIST[j], 'num_lot']
                                old_df = group_num_sequence[group_num_sequence['num_lot'] == old_lot].sort_index()
                                
                                new_lot = group_num_sequence.at[PACKING_INDEX_LIST[j+ 1], 'num_lot']
                                new_df = group_num_sequence[group_num_sequence['num_lot'] == new_lot].sort_index()
                                
                                get_length = min(len(old_df), len(new_df))
                                
                                if len(old_df) >= len(new_df):
                                    old_df = old_df[-get_length:]
                                else:
                                    new_df = new_df[-get_length:]
                                
                                print('old_df', old_df[['num_job', 'num_lot','operation']])
                                print('new_df', new_df[['num_job', 'num_lot','operation']])
                                temp_df = pd.DataFrame()
                                for new_row_idx, new_row in new_df.iterrows():
                                    for old_row_idx, old_row in old_df.iterrows():
                                        df = pd.DataFrame()
                                        df1 = pd.DataFrame()
                                        
                                        if new_row['operation'] == old_row['operation']:
                                            df = pd.DataFrame({new_row_idx : old_row})
                                            df = df.T
                                            #temp_df = temp_df.append(df)
                                            df1 = pd.DataFrame({old_row_idx : new_row})
                                            df1 = df1.T
                                            df = df.append(df1)
                                        temp_df = temp_df.append(df)
                                print('temp_df', temp_df[['num_job', 'num_lot','operation']])
                                group_num_sequence.update(temp_df)
                                group_part_job.update(temp_df)
                                observation.update(temp_df)
                                
                                #print('new', PACKING_INDEX_LIST[j], PACKING_INDEX_LIST[j + 1])    
                    observation.sort_index()
                            
                    return PACKING_INDEX_LIST, observation
            
            packing_index_list, observation = buble_sort(packing_index_list, group_num_sequence, observation)
            print('packing_index_list', packing_index_list)
            excel_writer(observation, file_output,'solution')
        
if __name__ == '__main__':
    print('Start!!')
    main = main(link, IOAM)