import pandas as pd
import numpy as np
from copy import deepcopy

input_link = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/input.xlsx'
link = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/output2.xlsx'
file_output = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/new_output.xlsx'
IOAM = 1
IJMM = 1
IJMM_rate = 10

print(pd.__version__)

def read_file(link):
    observation = pd.read_excel(link, sheet_name='best_solution')
    return observation


def sequence_import(input_link):
    sequence = pd.read_excel(input_link, sheet_name='sequence')
    return sequence

def excel_writer(obervation, file, sheet_name):
    writer = pd.ExcelWriter(file)
    obervation.to_excel(writer, sheet_name)
    writer.save()


def part_sequence(sequence, part, num_sequence):
    part_sequence = sequence[(sequence.part == part) & (sequence.num_sequence == num_sequence)]
    part_sequence = part_sequence.dropna(axis=1, how='any')
    part_sequence = part_sequence.values.flatten().tolist()[2:]
    return part_sequence

def main(link, IOAM):
    observation = read_file(link)
    
    if IJMM > np.random.uniform(0,1):
        sequence = sequence_import(input_link)
        
        def get_prev_row_operation(part_sequence, operation):
                operation_index = part_sequence.index(operation)
                if operation_index - 1 > 0:
                    prev_row_operation = part_sequence[operation_index - 1]
                else:
                    prev_row_operation = None
                return prev_row_operation

        def get_post_row_operation(part_sequence, operation):
                operation_index = part_sequence.index(operation)
                try:
                    post_row_operation = part_sequence[operation_index + 1]
                except:
                    post_row_operation = None
                return post_row_operation
        
        IJMM_pick_df = observation.sample(IJMM_rate,replace=False)
        IJMM_pick_df = IJMM_pick_df.sort_index()
        #print('IJMM_pick_df', IJMM_pick_df[['num_job', 'num_lot','operation']])
        
        for row_idx, row in IJMM_pick_df.iterrows():
            row_part = row['part']
            row_num_lot = row['num_lot']
            row_operation = row['operation']
            row_machine = row['machine']
            row_num_sequence = row['num_sequence']
            row_part_sequence = part_sequence(sequence, row_part, row_num_sequence)
            
            prev_row_operation = get_prev_row_operation(row_part_sequence, row_operation)
            post_row_operation = get_post_row_operation(row_part_sequence, row_operation)

            try:
                row_min_idx = observation.index[(observation.num_lot == row_num_lot)
                                                & (observation.operation == prev_row_operation)].tolist()[0]
            except:
                row_min_idx = 0
            try:
                row_max_idx = observation.index[(observation.num_lot == row_num_lot)
                                                & (observation.operation == post_row_operation)].tolist()[0]
            except:
                row_max_idx = len(observation)

            check_machine_df = observation.loc[(observation.machine == row_machine)
                                              & (observation.num_sequence == row_num_sequence)
                                              & (observation.index > row_min_idx)
                                              & (observation.index < row_max_idx)]
            # print('check_machine_df', check_machine_df[['num_job', 'num_lot','operation']])
            review_df = observation.loc[row_min_idx:row_max_idx, :]
            print('review_df', review_df[['num_job', 'num_lot','operation']])
            # print(row_min_idx)
            # print(row_max_idx)
            
            print('check_machine_df', check_machine_df[['num_job', 'part', 'machine', 'num_lot', 'operation']])
            df = pd.DataFrame(columns=['num_job',
                                                 'num_lot',
                                                 'part',
                                                 'operation',
                                                 'machine',
                                                 'num_sequence'], dtype=str)
            for check_row_idx, check_row in check_machine_df.iterrows():
                #print('check_row', check_row[['num_job', 'num_lot','operation']])
                check_row_part = check_row['part']
                check_row_num_sequence = check_row['num_sequence']
                check_row_num_lot = check_row['num_lot']
                check_row_part_sequence = part_sequence(sequence, check_row_part, check_row_num_sequence)
                check_row_operation = check_row['operation']

                check_prev_row_operation = get_prev_row_operation(check_row_part_sequence, check_row_operation)
                try:
                    check_prev_row_operation_idx = review_df.index[(review_df.num_lot == check_row_num_lot)
                                                & (review_df.operation == check_prev_row_operation)].tolist()[0]
                except:
                    check_prev_row_operation_idx = None
                
                check_post_row_operation = get_post_row_operation(check_row_part_sequence, check_row_operation)
                
                try:
                    check_post_row_operation_idx = review_df.index[(review_df.num_lot == check_row_num_lot)
                                                & (review_df.operation == check_post_row_operation)].tolist()[0]
                except:
                    check_post_row_operation_idx = None
                                
                if ((check_prev_row_operation_idx == None) and
                    (check_post_row_operation_idx == None)):
                    df = df.append(check_row)
            
            try:
                print('df', df[['num_job', 'part', 'machine', 'num_lot', 'operation']])
            except:
                pass
            if (len(df) >= 2):
                print('lod', df[['num_job', 'part', 'machine', 'num_lot', 'operation']])
                df_idx = df.index.tolist()
                df['num_job'] = df['num_job'].astype(int)
                df = df.sort_values(['num_job'])
                df.index = df_idx
                print('new',df[['num_job', 'part', 'machine', 'num_lot', 'operation']])
            observation.update(df)
    excel_writer(observation, r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/IJMM.xlsx','IJMM')


    if IOAM > np.random.uniform(0,1):
        part_list = sorted(observation['part'].unique())
        # observation_list = observation.index.tolist()
        leftover_df = pd.DataFrame()
        # print('observation_list', observation_list)
        temp_df = pd.DataFrame()
        for part in part_list:
            group_part_job = observation.loc[observation.part == part][:]
            num_sequence_list = sorted(group_part_job['num_sequence'].unique())
            for num_sequence in num_sequence_list:
                group_num_sequence = group_part_job.loc[group_part_job.num_sequence == num_sequence][:]
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
                    print(len(old_df), len(new_df))
                    if len(old_df) >= len(new_df):
                        old_df = old_df[-get_length:]
                        print('A len new',new_df[['num_job', 'num_lot','operation']])
                        print('A len old',old_df[['num_job', 'num_lot','operation']])

                    else:
                        leftover_df = leftover_df.append(new_df[:- get_length])
                        new_df = new_df[-get_length:]
                        print('B len new',new_df[['num_job', 'num_lot','operation']])
                        print('B len old',old_df[['num_job', 'num_lot','operation']])
                        

                    new_df.index = old_df.index.tolist() 

                    temp_df = temp_df.append(new_df)
        print('B temp', temp_df[['num_job', 'num_lot','operation']])


            #print('temp', temp_df[['num_job', 'num_lot','operation']])
            #excel_writer(temp_df, r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/%s.xlsx' %part, part)

        #temp_full = temp_full.append(leftover_df)
        excel_writer(leftover_df, r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/leftover.xlsx','leftover')
        
        excel_writer(temp_df, r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/temp_df.xlsx','temp_df')
        temp_full = temp_df.append(leftover_df)
        temp_full = temp_full.sort_index()
        temp_full = temp_full.reset_index(drop=True)
        print(len(temp_full))
        excel_writer(temp_full, r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/temp_full.xlsx','temp_full')

    lot_list = temp_full['num_lot'].unique()
    for lot in lot_list:
        num = len(temp_full[temp_full['num_lot']==lot])
        print('Lot# %s has %s lot' %(lot, num))
                    
    excel_writer(observation, file_output,'solution')
    print()
    

if __name__ == '__main__':
    print('Start!!')
    main = main(link, IOAM)