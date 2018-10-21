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
        leftover_df = pd.DataFrame()
        check2 = pd.DataFrame()
        print('observation_list', observation_list)
        temp_df = pd.DataFrame()
        for part in part_list:
            group_part_job = observation.loc[observation.part == part].copy()
            num_sequence_list = sorted(group_part_job['num_sequence'].unique())
            for num_sequence in num_sequence_list:
                group_num_sequence = group_part_job.loc[group_part_job.num_sequence == num_sequence]
                packing_index_list = group_num_sequence.index[group_num_sequence.operation == 'packing'].tolist()
                
                def quick_sort(PACKING_INDEX_LIST, group_num_sequence):
                    '''Using code from github.com/TheAlgorithms/Python/blob/master/sorts/quick_sort.py'''
                    length = len(PACKING_INDEX_LIST)
                    if length <= 1:
                        return PACKING_INDEX_LIST
                    else:
                        PIVOT = PACKING_INDEX_LIST[0]
                        GREATER = [element for element in PACKING_INDEX_LIST[1:] if group_num_sequence.at[element, 'num_job'] >= group_num_sequence.at[PIVOT, 'num_job']]
                        LESSER = [element for element in PACKING_INDEX_LIST[1:] if group_num_sequence.at[element, 'num_job'] < group_num_sequence.at[PIVOT, 'num_job']]
                        return quick_sort(LESSER, group_num_sequence) + [PIVOT] +quick_sort(GREATER, group_num_sequence)
                    
                new_packing_index_list = quick_sort(packing_index_list, group_num_sequence)
                print('old', packing_index_list)
                print('new', new_packing_index_list)
                
                for idx in range(len(packing_index_list)):
                    new_pack_idx = new_packing_index_list[idx]
                    old_pack_idx = packing_index_list[idx]

                    print(old_pack_idx, new_pack_idx)
                    
                    old_lot = group_num_sequence.at[old_pack_idx, 'num_lot']
                    old_df = deepcopy(group_num_sequence[group_num_sequence['num_lot'] == old_lot].sort_index())
                    
                    new_lot = group_num_sequence.at[new_pack_idx, 'num_lot']
                    new_df = deepcopy(group_num_sequence[group_num_sequence['num_lot'] == new_lot].sort_index())
                    get_length = min(len(old_df), len(new_df))
                    if len(old_df) >= len(new_df):
                        old_df = old_df[-get_length:]
                        print('A len new',new_df[['num_job', 'num_lot','operation']])
                        print('A len old',old_df[['num_job', 'num_lot','operation']])
                        
                        for new_row_idx, new_row in new_df.iterrows():
                            new_opt = new_row['operation']
                            old_row_idx = old_df.index[old_df['operation'] == new_opt].tolist()[0]
                            df = pd.DataFrame({old_row_idx : new_row})
                            df = df.T
                            temp_df = temp_df.append(df)
                    
                    else:
                        leftover_df = leftover_df.append(new_df[:- get_length])
                        new_df = new_df[-get_length:]
                        print('B len new',new_df[['num_job', 'num_lot','operation']])
                        print('B len old',old_df[['num_job', 'num_lot','operation']])
                        
                        for new_row_idx, new_row in new_df.iterrows():
                            new_opt = new_row['operation']
                            old_row_idx = old_df.index[old_df['operation'] == new_opt].tolist()[0]
                            df = pd.DataFrame({old_row_idx : new_row})
                            df = df.T
                            temp_df = temp_df.append(df)
        print('B temp', temp_df[['num_job', 'num_lot','operation']])


            #print('temp', temp_df[['num_job', 'num_lot','operation']])
            #excel_writer(temp_df, r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/%s.xlsx' %part, part)

        #temp_full = temp_full.append(leftover_df)
        excel_writer(leftover_df, r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/leftover.xlsx','leftover')
        
        excel_writer(temp_df, r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/temp_df.xlsx','temp_df')
        temp_full = temp_df.append(leftover_df)
        temp_full = temp_full.sort_index()
        temp_full = temp_full.reset_index(drop=True)
        
        excel_writer(temp_full, r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/temp_full.xlsx','temp_full')
        
        '''
        delete_list =[]
        for trow_idx, trow in temp_full.iterrows():
            for row_idx, row in observation.iterrows():
                if trow.equals(row):
                    print(row_idx)
                    delete_list.append(row_idx)
                    
        print(delete_list)
        observation = observation.drop(observation.index[delete_list])
            #observation.update(temp_df)
        print('observation_list', observation_list)
        '''
    lot_list = temp_full['num_lot'].unique()
    for lot in lot_list:
        num = len(temp_full[temp_full['num_lot']==lot])
        print('Lot# %s has %s lot' %(lot, num))
                    
    excel_writer(observation, file_output,'solution')
    

if __name__ == '__main__':
    print('Start!!')
    main = main(link, IOAM)