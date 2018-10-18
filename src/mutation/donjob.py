import pandas as pd
import numpy as np


link_criteria = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/input.xlsx'
link = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/output2.xlsx'
file_output = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/new_output.xlsx'
IOAM = 1
def read_file(link):
    observation = pd.read_excel(link, sheet_name='best_solution')
    return observation

def read_sequence(link):
    sequence = pd.read_excel(link, sheet_name='sequence')
    return sequence

def part_sequence(self, part, num_sequence):
        part_sequence = self.sequence[(self.sequence.part == part) & (self.sequence.num_sequence == num_sequence)]
        part_sequence = part_sequence.dropna(axis=1, how='any')
        part_sequence = part_sequence.values.flatten().tolist()[2:]
        return part_sequence

def excel_writer(obervation, file):
    writer = pd.ExcelWriter(file)
    obervation.to_excel(writer, sheet_name='new_solution')
    writer.save()

def main(link, IOAM):
    observation = read_file(link)

    if IOAM > np.random.uniform(0,1):
        part_list = sorted(observation['part'].unique())
        for part in part_list:
            group_part_job = observation.loc[observation.part == part]
            num_sequence_list = sorted(group_part_job['num_sequence'].unique())
            for num_sequence in num_sequence_list:
                group_num_sequence = group_part_job.loc[group_part_job.num_sequence == num_sequence]
                packing_index_list = group_num_sequence.index[group_num_sequence.operation == 'packing'].tolist()
                print(packing_index_list)
                def quick_sort(PACKING_INDEX_LIST, group_num_sequence):
                    '''Using code from github.com/TheAlgorithms/Python/blob/master/sorts/quick_sort.py'''
                    length = len(PACKING_INDEX_LIST)
                    if length <= 1:
                        return PACKING_INDEX_LIST
                    else:
                        PIVOT = PACKING_INDEX_LIST[0]
                        GREATER = [element for element in PACKING_INDEX_LIST[1:] if group_num_sequence.at[element, 'num_job'] > group_num_sequence.at[PIVOT, 'num_job']]
                        LESSER = [element for element in PACKING_INDEX_LIST[1:] if group_num_sequence.at[element, 'num_job'] <= group_num_sequence.at[PIVOT, 'num_job']]
                        return quick_sort(LESSER, group_num_sequence) + [PIVOT] +quick_sort(GREATER, group_num_sequence)
                    
                new_packing_index_list = quick_sort(packing_index_list, group_num_sequence)
                print('new', new_packing_index_list)
                for i in range(len(new_packing_index_list)):
                    temp_df = pd.DataFrame()
                    new_idx = new_packing_index_list[i]
                    forward_lot = group_num_sequence.at[new_idx, 'num_lot']
                    forward_df = group_num_sequence[group_num_sequence.num_lot == forward_lot].copy()
                    
                    old_idx = packing_index_list[i]
                    delete_lot = group_num_sequence.at[old_idx, 'num_lot']
                    delete_df = group_num_sequence[group_num_sequence.num_lot == delete_lot].copy()
                    
                    for fwd_row_idx, fwd_row in forward_df.iterrows():
                        for bwd_row_idx, bwd_row in delete_df.iterrows():
                            if fwd_row['operation'] == bwd_row['operation']:
                                fwd_row['start_time'] = bwd_row['start_time']
                                fwd_row['completion_time'] = bwd_row['completion_time']
                                new_df = pd.DataFrame({bwd_row_idx : fwd_row})
                                new_df = new_df.T
                                temp_df = temp_df.append(new_df)
                    print('temp', temp_df)

                    observation.update(temp_df)
                    
                    
    excel_writer(observation, file_output)

                                            # print('forward index len %s, and backward index len %s' %(len(forward_lot_index_list), len(backward_lot_index_list)))

if __name__ == '__main__':
    print('Start!!')
    main = main(link, IOAM)