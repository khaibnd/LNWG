import pandas as pd
import numpy as np
import datetime as dt
import calendar
import math



class CapacityChart():

    def __init__(self, output, raw_input):
        self.output = output
        self.raw_input = raw_input
        
    
    def analyst_data(self, output, raw_input):
        '''Build chart to compare total capacity vs load plan'''
        load_plan = pd.read_excel(output, sheet_name='best_solution')
        machine = pd.read_excel(raw_input, sheet_name='machine')
        machine_criteria = pd.read_excel(raw_input, sheet_name='machine_criteria')
        load_time = pd.DataFrame(columns=['date', 'operation', 'machine', 'run_time', 'free_time', 'total_time'])
        
        start_day = min(load_plan['calendar_start_time'].min().date(), pd.to_datetime('now').date())
        year = load_plan['calendar_completion_time'].max().date().year
        month = load_plan['calendar_completion_time'].max().date().month
        day = calendar.monthrange(year, month)[1]
        end_day = pd.datetime(year, month, day).date()
        
        operation_list = machine.columns.values
        print(operation_list)
        machine_list = machine_criteria['machine'].tolist()
        
        def get_row_run_time(row):
            if row['calendar_completion_time'].dt.date == row['calendar_start_time'].dt.date:
                run_time = row['calendar_completion_time'] - row['calendar_start_time']
                return 
            elif row['calendar_completion_time'].dt.date < row['calendar_start_time'].dt.date:
                return row['calendar_completion_time'].dt.date - row['calendar_start_time'].dt.date
            else:
                return row['calendar_completion_time'] - row['calendar_start_time']
        
        
        def get_run_time(load_plan, date, operation, machine):
            try:
                run_time_df = load_plan.loc[(load_plan.operation == operation) &
                                            (load_plan.machine == machine) &
                                            ((load_plan.calendar_start_time.dt.date==date) | (load_plan.calendar_completion_time.dt.date==date))]
                print(run_time_df)

                run_time_df['run_time'] = run_time_df.apply(lambda row: (row.calendar_completion_time - row.calendar_start_time), axis=1)
                run_time_df['run_time'] = pd.to_datetime(run_time_df['run_time'])
                run_time_df['run_time'] = run_time_df['run_time'].dt.hour + run_time_df['run_time'].dt.minute/60

                run_time = run_time_df['run_time'].sum()
                print(run_time)
            except:
                run_time = 0
            return run_time                                            
            
        def datetime_range(start=None, end=None):
            num_days = (end_day - start_day).days
            for i in range(num_days + 1):
                yield start + dt.timedelta(days=i)
        date_list = list(datetime_range(start_day, end_day))
        for _, date in enumerate(date_list):
            for _, operation in enumerate(operation_list):
                for _, machine in enumerate(machine_list):
                    if get_run_time(load_plan, date, operation, machine) >0:
                        row = pd.DataFrame([[date,
                                             operation,
                                             machine,
                                             get_run_time(load_plan, date, operation, machine),
                                             24 - get_run_time(load_plan, date, operation, machine),
                                             24]], columns=['date', 'operation', 'machine', 'run_time', 'free_time', 'total_time'])
        
                        load_time = load_time.append(row)
        #load_time.drop('index')
        load_time.reset_index(inplace=True)
        
        return load_time.loc[load_time.run_time > 0]
    
    def bokeh_capacity_week_chart(self, src):
        '''Build chart to compare total capacity vs load plan'''
        output = pd.read_excel(src, sheet_name='best solution')
    
        return 1
    
    def bokeh_capacity_month_chart(self, src):
        '''Build chart to compare total capacity vs load plan'''
        output = pd.read_excel(src, sheet_name='best solution')
    
    
    def bokeh_capacity_operations_chart(self, src):
        '''Build chart to compare total capacity vs load plan'''
        output = pd.read_excel(src, sheet_name='best solution')



if __name__ == '__main__':
    print("LNWG Capacity chart")
    output = r'/Users/khaibnd/github-repositories/LNWG/src/data/output3.xlsx'
    raw_input = r'/Users/khaibnd/github-repositories/LNWG/src/data/input.xlsx'
    CapacityChart = CapacityChart(output, raw_input)
    print(CapacityChart.analyst_data(output, raw_input))
    