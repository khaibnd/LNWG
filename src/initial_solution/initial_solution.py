'''Generate GA initial solution [lot, operation]'''
import random
from copy import deepcopy
import sta
import pandas as pd
import numpy as np

global Stop


class InitialSolution():
    ''''''

    def __init__(self, parameter, demand, wip, criteria, machine, sequence):
        self.parameter = parameter
        self.demand = demand
        self.wip = wip
        self.criteria = criteria
        self.machine = machine
        self.sequence = sequence
        self.sequence_type = ['seq_1', 'seq_2', 'seq_3']

    def generate_num_lot_required(self):
        '''List of number of lot for each demand line'''
        num_lot_required = []
        for demand_index in self.demand.index:
            num_lot_required.append(int(round(self.demand.loc[demand_index, 'demand']\
                / self.criteria.loc[self.criteria.part == \
                    self.demand.loc[demand_index, 'part']]['yield']
                / self.criteria.loc[self.criteria.part == \
                    self.demand.loc[demand_index, 'part']]['dicing_lotsize'])))
        return num_lot_required

    def generate_selected_job_list(self):
        '''Randomly selection job list from list of wip'''
        run_part = self.demand['part'].unique()

        def available_list(part, sequence):
            return self.wip[(self.wip.part == part)
                            & (self.wip.num_sequence == sequence)]['job_num'].tolist()
        
        available_job = pd.DataFrame(index=run_part, columns=self.sequence_type)
        for part in run_part:
            for num_sequence in self.sequence_type:
                try:
                    available_job.loc[part, num_sequence] = available_list(part, num_sequence)
                except:
                    available_job.loc[part, num_sequence] = None
        return  available_job

    def generate_machine_list(self, sequence, operation):
        '''Randomly selection machine for operation from list of num_machine'''
        return sta(self.machine[self.machine.num_sequence == sequence][operation].values[0])

    def part_sequence(self, part, num_sequence):
        part_sequence = self.sequence[(self.sequence.part == part)
                                      & (self.sequence.num_sequence == num_sequence)]
        part_sequence = part_sequence.dropna(axis=1, how='any')
        part_sequence = part_sequence.values.flatten().tolist()[2:]
        return part_sequence

    def generate_num_lot_each_sequence(self):
        num_lot_required = self.generate_num_lot_required()
        demand_index = self.demand.index.values.tolist()
        num_lot_each_sequence = pd.DataFrame(index=demand_index, columns=self.sequence_type)
        for demand in demand_index:
            for sequence in self.sequence_type:
                try:
                    num_lot_each_sequence.loc[demand, sequence] = int(num_lot_required[demand] * self.demand.at[demand, sequence])
                except:
                    num_lot_each_sequence.loc[demand, sequence] = np.nan
        return num_lot_each_sequence

    def observation_sequence_sort(self, initial_observation):
        observation = pd.DataFrame()
        for lot in initial_observation['num_lot'].unique():
            # Get current part of lot
            part = initial_observation[
                initial_observation['num_lot'] == lot]['part'].values[0]
            sequence = initial_observation[
                initial_observation['num_lot'] == lot]['num_sequence'].values[0]
            # Get part production sequence
            part_sequence = InitialSolution.part_sequence(self, part, sequence)
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
            observation = observation.append(sub_df_per_lot)
            # Sort Dataframe by accending index
        observation = observation.sort_index()
        return observation

    def generate_df_required_job_num(self):
        '''Sequence job to dataframe'''
        selected_job = []
        selected_job_num_list = []
        job_num_part_list = []
        job_num_operation_list = []
        job_num_machine_list = []
        sequence_list = []
        selected_job_list = self.generate_selected_job_list()
        num_lot_each_sequence = self.generate_num_lot_each_sequence()
        i = 0
        for demand_index, num_sequence_series in num_lot_each_sequence.iterrows():
            num_sequence_series = num_sequence_series.dropna()
            part = self.demand.loc[demand_index, 'part']
            for num_sequence, total_lot in num_sequence_series.iteritems():
                part_sequence = self.part_sequence(part, num_sequence)
                available_jobs = selected_job_list.loc[part, num_sequence]
                for _ in range(total_lot):
                    # lot num select
                    if available_jobs != [] and available_jobs != None:
                        selected_job_list_sequence_index = []
                        for j in available_jobs:
                            curent_job_operation = self.wip.loc[self.wip.job_num == j, 'operation'].values[0]

                            selected_job_list_sequence_index.append(part_sequence.index(curent_job_operation))
                        max_value = max(selected_job_list_sequence_index)
                        max_value_index = selected_job_list_sequence_index.index(max_value)
                        selected_job_num = selected_job_list.at[part, num_sequence][max_value_index]
                        available_jobs.remove(selected_job_num)
                        # selected_job_list.loc[part, num_sequence].remove(selected_job_num)
                        selected_job_num_list.append(selected_job_num)
                        # operation select
                        job_num_operation = self.wip.loc[self.wip.job_num == selected_job_num, 'operation'].values[0]
                        job_num_operation_list.append(job_num_operation)
                    else:
                        selected_job_num_list.append('New lot %s' % i)
                        i += 1
                        job_num_operation = 'release'
                        job_num_operation_list.append(job_num_operation)
                    # job select
                    selected_job.append(demand_index)        
                    # part select
                    job_num_part_list.append(part)
                    
                    # Machine Sellect
                    num_machine = self.generate_machine_list(num_sequence, job_num_operation)
                    
                    job_num_machine = random.choice(num_machine)
                    job_num_machine_list.append(job_num_machine)
                    
                    # Sequence
                    sequence_list.append(num_sequence)

        data = {'num_job': selected_job, 'num_lot': selected_job_num_list,
                'part': job_num_part_list, 'operation': job_num_operation_list,
                'machine': job_num_machine_list, 'num_sequence': sequence_list}
        df_required_job_num = pd.DataFrame.from_dict(data)
        return df_required_job_num
    
    def generate_df_full_sequence_job_num(self):
        '''Adding processor sequence job to dataframe'''
        df_full_sequence_job_num = self.generate_df_required_job_num()
        for row_index, row in df_full_sequence_job_num.iterrows():
            part = row['part']
            sequence = row['num_sequence']
            part_sequence = self.part_sequence(part, sequence)
                
            current_operation = row['operation']
            current_operation_sequence_index = part_sequence.index(current_operation)

            while current_operation_sequence_index < len(part_sequence) - 1:
                next_operation = part_sequence[current_operation_sequence_index + 1]

                num_machine = self.generate_machine_list(sequence, next_operation)
                job_num_machine = random.choice(num_machine)

                add_row = pd.DataFrame([[row['num_job'],
                                         row['num_lot'],
                                         part,
                                         next_operation,
                                         job_num_machine,
                                         sequence]],
                                        columns=['num_job',
                                                 'num_lot',
                                                 'part',
                                                 'operation',
                                                 'machine',
                                                 'num_sequence'], dtype=str)
                df_full_sequence_job_num = \
                    df_full_sequence_job_num.append(add_row,
                                                    ignore_index=True)

                current_operation_sequence_index += 1
        df_full_sequence_job_num = df_full_sequence_job_num.astype('str')
        return df_full_sequence_job_num

    def generate_initial_population(self):
        '''Initial population generation'''
        population_size = int(
            self.parameter[self.parameter['name'] == 'population_size']['value'])
        initial_population = {}
        initial_observation = InitialSolution.generate_df_full_sequence_job_num(self)

        for num_observation in range(population_size):
            observation = pd.DataFrame()
            initial_observation = initial_observation.sample(frac=1).reset_index(drop=True)
            observation = InitialSolution.observation_sequence_sort(self, initial_observation)

            initial_population[num_observation] = observation
        return initial_population
