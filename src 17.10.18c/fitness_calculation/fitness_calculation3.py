'''Calculate production leadtime per observation'''
import math
from src.initial_solution.initial_solution import InitialSolution


class FitnessCalculation():
    '''Measure invidial demand job completion time and Total Weighed Tardiness'''
    def __init__(self, observation, sequence, machine_lotsize):
        self.observation = observation
        self.sequence = sequence
        self.machine_lotsize = machine_lotsize


    def prev_part_sequence_operation(self, part, operation, num_sequence):
        part_sequence = InitialSolution.part_sequence(self, part, num_sequence)
        operation_index = part_sequence.index(operation)
        try:
            prev_part_sequence_operation = part_sequence[operation_index - 1]
        except:
            prev_part_sequence_operation = None
        return prev_part_sequence_operation


    def calculate_job_processing_time(self, part, operation, num_sequence):
        '''Completion time each num_lot(= batch processing time)'''
        def line_index(num_sequence, part):
            line_index = self.processing_time.index[(self.processing_time['num_sequence'] == num_sequence)
                                              & (self.processing_time['part'] == part)].tolist()
            # Get the smallest index
            line_index = line_index[0]
            return line_index
        # print('line', part, operation, num_sequence)
        job_processing_time = self.processing_time.at[line_index(num_sequence, part), operation]
        # print('job_processing_time', part, operation, num_sequence, job_processing_time)
        return job_processing_time


    def completion_time_with_daily_work_hour(self,
                                             row_index,
                                             job_processing_time,
                                             machine,
                                             completion_time,
                                             prev_machine_completion_time):
        machine_daily_work_hour = self.machine_criteria.loc[self.machine_criteria.machine == machine, 'work_hour'].values[0]
        num_day = math.floor((completion_time[row_index] - prev_machine_completion_time[machine])/machine_daily_work_hour)
        free_hour = completion_time[row_index] - prev_machine_completion_time[machine] - job_processing_time
        if num_day > 0 and free_hour == 0:
            prev_machine_completion_time[machine] = completion_time[row_index] + num_day * (24 - machine_daily_work_hour)
        elif num_day > 0 and free_hour <= num_day * (24 - machine_daily_work_hour):
            prev_machine_completion_time[machine] = completion_time[row_index] + num_day * (24 - machine_daily_work_hour) - free_hour
        else:
            prev_machine_completion_time[machine] = completion_time[row_index]
        return prev_machine_completion_time[machine]

 
    def calculate_finished_time(self, observation):
        '''Chua xem xet may tsk(Gop 4 lot)'''
        demand_job_list = observation['num_job'].unique()
        num_lot_list = observation['num_lot'].unique()
        machine_list = observation['machine'].unique()

        num_assign_max_lotsize = {job: 0
                                   for job in observation.index}
        num_assign_min_lotsize = {job: 0
                                   for job in observation.index}
        completion_time = {job: 0
                           for job in observation.index}
        prev_demand_job_completion_time = {num_job:0
                                           for num_job in demand_job_list}
        prev_machine_completion_time = {machine: 0
                                        for machine in machine_list}
        prev_operation_completion_time = {num_lot: 0
                                          for num_lot in num_lot_list}
        # print(machine_assign_time, prev_machine_completion_time, prev_operation_completion_time)
        for row_index, row in observation.iterrows():
            num_job = row['num_job']
            num_lot = row['num_lot']
            part = row['part']
            operation = row['operation']
            machine = row['machine']
            num_sequence = row['num_sequence']
            try:
                job_processing_time = FitnessCalculation.calculate_job_processing_time(self,
                                                                                       part,
                                                                                       operation,
                                                                                       num_sequence)
            except:
                print('Wrong processing time!')
                print('part: %s, operation: %s, num_sequence: %s' %(part, operation, num_sequence))


            def ga_calculation(self):
                # Step 5.1: First assignment to machine m and operation
                if (prev_machine_completion_time[machine] == 0 and
                    prev_operation_completion_time[num_lot] == 0):
                    completion_time[row_index] = job_processing_time
                    prev_demand_job_completion_time[num_job] = max(prev_demand_job_completion_time[num_job],
                                                                   completion_time[row_index])
                    prev_operation_completion_time[num_lot] = completion_time[row_index]
                    prev_machine_completion_time[machine] = FitnessCalculation.completion_time_with_daily_work_hour(self,
                                                                                                                    row_index,
                                                                                                                    job_processing_time,
                                                                                                                    machine,
                                                                                                                    completion_time,
                                                                                                                    prev_machine_completion_time)
    
                # Step 5.2: First assignment to machine m and operation
                elif ((prev_machine_completion_time[machine] == 0)
                    and (prev_operation_completion_time[num_lot] > 0)):
                    completion_time[row_index] = prev_operation_completion_time[num_lot] + job_processing_time
                    prev_demand_job_completion_time[num_job] = max(prev_demand_job_completion_time[num_job],
                                                                   completion_time[row_index])
                    prev_operation_completion_time[num_lot] = completion_time[row_index]
                    
                    prev_machine_completion_time[machine] = FitnessCalculation.completion_time_with_daily_work_hour(self,
                                                                                                                    row_index,
                                                                                                                    job_processing_time,
                                                                                                                    machine,
                                                                                                                    completion_time,
                                                                                                                    prev_machine_completion_time)
    
                # Step 5.3: First operation in sequence, and machine m assignment > 1
                elif ((prev_machine_completion_time[machine] > 0)
                    and (prev_operation_completion_time[num_lot] == 0)):
                    completion_time[row_index] = prev_machine_completion_time[machine] + job_processing_time
                    prev_demand_job_completion_time[num_job] = max(prev_demand_job_completion_time[num_job],
                                                                   completion_time[row_index])
                    prev_operation_completion_time[num_lot] = completion_time[row_index]
                    
                    prev_machine_completion_time[machine] = FitnessCalculation.completion_time_with_daily_work_hour(self,
                                                                                                                    row_index,
                                                                                                                    job_processing_time,
                                                                                                                    machine,
                                                                                                                    completion_time,
                                                                                                                    prev_machine_completion_time)
    
                # Step 5.4: Not the first operation, and machine m assignment > 1
                else:
                    completion_time[row_index] = max(prev_machine_completion_time[machine],
                                               prev_operation_completion_time[num_lot]) + job_processing_time
                    prev_demand_job_completion_time[num_job] = max(prev_demand_job_completion_time[num_job],
                                                                   completion_time[row_index])
                    prev_operation_completion_time[num_lot] = completion_time[row_index]
   
                    prev_machine_completion_time[machine] = FitnessCalculation.completion_time_with_daily_work_hour(self,
                                                                                                                    row_index,
                                                                                                                    job_processing_time,
                                                                                                                    machine,
                                                                                                                    completion_time,
                                                                                                                    prev_machine_completion_time)
    
                        
                return completion_time[row_index], prev_demand_job_completion_time[num_job], prev_operation_completion_time[num_lot]

            def line_index(machine, num_sequence):
                try:
                    line_index = self.machine_lotsize.index[(self.machine_lotsize.machine == machine)
                                                              & (self.machine_lotsize.num_sequence == num_sequence)].tolist()
                    line_index = line_index[0]
                except(IndexError):
                    print('IndexError: list index out of range')
                    print('Combination %s and %s is not in the input file' %(machine, num_sequence))
                return line_index


            max_lotsize = int(self.machine_lotsize.at[line_index(machine, num_sequence), 'max_lotsize'])
            min_lotsize = int(self.machine_lotsize.at[line_index(machine, num_sequence), 'min_lotsize'])
            
            if (num_assign_max_lotsize[row_index] == 0) and (num_assign_min_lotsize[row_index] == 0):
                assign_available = True
            else:
                assign_available = False

            # Step 6.1: Maximum and minimum lot per run = 1
            if (max_lotsize == 1) and (min_lotsize == 1):
                completion_time[row_index],\
                prev_demand_job_completion_time[num_job],\
                prev_operation_completion_time[num_lot] = ga_calculation(self)


            # Step 6.2: maximum lotsize > 1 and minimum lotsize = 1
            elif (min_lotsize == 1) and (assign_available == True):
                lot_size = 1
                completion_time[row_index],\
                prev_demand_job_completion_time[num_job],\
                prev_operation_completion_time[num_lot] = ga_calculation(self)
                succeed_observation = observation.loc[row_index+1:,:]

                # Review groupability of precedence genes
                for group_row_index, group_row in succeed_observation.iterrows():
                    # Step 6.2.1 : Check total lot per run in machine should lower than maximum.
                    if lot_size < max_lotsize:
                        
                        def groupable_condition(self):
                            if (group_row['operation'] == 'coat' and
                                group_row['machine'] == machine):
                                coat_design = self.criteria.loc[self.criteria.part == part, 'coat_design'].values[0]
                                group_part = group_row['part']
                                group_coat_design = self.criteria.loc[self.criteria.part == group_part, 'coat_design'].values[0]
                                if group_coat_design == coat_design:
                                    return True
                                else:
                                    return False
                            elif (group_row['part'] == part and
                                  group_row['operation'] == operation and
                                  group_row['operation'] != 'coat' and
                                  group_row['machine'] == machine and
                                  group_row['num_sequence'] == num_sequence):
                                return True
                            else:
                                return False
                            
                        prev_operation = FitnessCalculation.prev_part_sequence_operation(self,
                                                                                         part,
                                                                                         operation,
                                                                                         num_sequence)

                        if (groupable_condition(self) == True):

                            # Step 6.2.1.2.1: The group index is first operation of sequence
                            if prev_operation == None:
                                print('First operation')
                                group_row_num_lot = group_row['num_lot']
                                group_num_job = group_row['num_job']
                                completion_time[group_row_index] = completion_time[row_index]
                                prev_demand_job_completion_time[group_num_job] = completion_time[group_row_index]
                                prev_operation_completion_time[group_row_num_lot] = completion_time[group_row_index]
                                lot_size +=1
                                num_assign_max_lotsize[group_row_index] +=1
                            # Step 6.2.1.2.1: The group index is not first operation of sequence    
                            else:
                                prev_group_observation = observation.loc[:row_index-1,:]
                                try:
                                    group_row_num_lot = group_row['num_lot']
                                    prev_row_index = prev_group_observation.index[(prev_group_observation['num_lot'] == group_row_num_lot)
                                                                        & (prev_group_observation['operation'] == prev_operation)].tolist()[0]
                                except:
                                    prev_row_index = None

                                # Step 6.2.1.2.1.1: Previous operation of observation exists
                                if prev_row_index != None:

                                    # Step 6.2.1.2.1.1.1: Job of Previous operation finished BEFORE group job start
                                    if completion_time[prev_row_index] < completion_time[row_index]:
                                        group_row_num_lot = group_row['num_lot']
                                        group_num_job = group_row['num_job']
                                        completion_time[group_row_index] = completion_time[row_index]
                                        prev_demand_job_completion_time[group_num_job] = completion_time[group_row_index]
                                        prev_operation_completion_time[group_row_num_lot] = completion_time[group_row_index]
                                        lot_size +=1
                                        num_assign_max_lotsize[group_row_index] +=1

                                    # Step 6.2.1.2.1.1.1: Job of Previous operation finished AFTER group job start
                                    else:
                                        try:
                                            group_row_num_lot = group_row['num_lot']
                                            prev_row_index = observation.index[(observation['num_lot'] == group_row_num_lot)
                                                                                & (observation['operation'] == prev_operation)].tolist()[0]
                                        except:
                                            prev_row_index = None
        
                                        # Step 6.2.1.2.1.1: Previous operation of observation exists
                                        if prev_row_index == None:
        
                                            # Step 6.2.1.2.1.1.1: Job of Previous operation finished BEFORE group job start
                                            if completion_time[prev_row_index] < completion_time[row_index]:
                                                group_row_num_lot = group_row['num_lot']
                                                group_num_job = group_row['num_job']
                                                completion_time[group_row_index] = completion_time[row_index]
                                                prev_demand_job_completion_time[group_num_job] = completion_time[group_row_index]
                                                prev_operation_completion_time[group_row_num_lot] = completion_time[group_row_index]
                                                lot_size +=1
                                                num_assign_max_lotsize[group_row_index] +=1

                        # Step 6.2.1.3: Operation canot group together
                        else:
                            pass

                    # Step 6.2.1 : Total lot per run in machine over maximum.
                    else:
                        pass
                
            elif (max_lotsize == 1) and (assign_available == True):
                print('min lot size > 1')

            elif (assign_available == True):
                print('max and min lot size > 1')
            else:
                pass


        return prev_demand_job_completion_time, prev_operation_completion_time, completion_time

    def calculate_makespan(self, observation):
        '''Calculate Demand total makespan'''
        prev_operation_completion_time, _, __ = FitnessCalculation.calculate_finished_time(self, observation)
        makespan = max(prev_operation_completion_time.values())
        return round(makespan, 1)
    
    def calculate_weighted_tardiness(self, observation):
        '''Calculate Demand total Weight Tardiness'''
        weighted_tardiness = 0
        demand_weight = self.demand['weight'].tolist()
        lead_time = self.demand['lead_time'].tolist()
        prev_demand_job_completion_time,_, __ = FitnessCalculation.calculate_finished_time(self, observation)
        for key in prev_demand_job_completion_time:
            weighted_tardiness += (prev_demand_job_completion_time[key] - lead_time[key] * 24) * demand_weight[key]
        return round(weighted_tardiness, 1)

if __name__=='__main__':
    fitness = FitnessCalculation()
    tardiness = fitness.calculate_weighted_tardiness(self, self.observation)
