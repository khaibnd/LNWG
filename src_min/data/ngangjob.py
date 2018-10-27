import pandas as pd
import numpy as np


input_link = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/input.xlsx'
link = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/output2.xlsx'
file_output = r'/Users/khaibnd/eclipse-workspace/LNWG4/src/data/new_output.xlsx'
IJMM = 1
IJMM_rate = 1


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


def main(link, input_link, IJMM, IJMM_rate):
    observation = read_file(link)
    # Intelligent Job Selection Mutation
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
        print(IJMM_pick_df)
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
            review_df = observation[row_min_idx:row_max_idx]
            print(row_min_idx)
            print(row_max_idx)
            
            print('check_machine_df', check_machine_df[['num_job', 'part', 'machine', 'num_lot', 'operation']])
            df = pd.DataFrame()
            for check_row_idx, check_row in check_machine_df.iterrows():
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
            
            if (len(df) >= 2):
                print(df[['num_job', 'part', 'machine', 'num_lot', 'operation']])
                df_idx = df.index.tolist()
                df = df.sort_values(['num_job'])
                df['num_job'] = df['num_job'].astype(int)
                df.index = df_idx
                print('new',df[['num_job', 'part', 'machine', 'num_lot', 'operation']])
            observation.update(df)
    excel_writer(observation, file_output, sheet_name='IJMM')
    
    return observation

if __name__ == '__main__':
    print('Start!!')
    main(link, input_link, IJMM, IJMM_rate)