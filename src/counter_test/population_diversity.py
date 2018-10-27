'''Kiem tra tinh da dang cua gene'''
import pandas as pd

class DiversityCheck():
    
    def __init__(self, population_dict):
        self.population_dict = population_dict
        
    def check_population_diversity(self):
        
        gene_idx_in_pop = self.population_dict[0].copy()
        
        for indiv_idx, indiv in iter(self.population_dict.items()):
            new_col = []
            for ini_indiv_idx, ini_indiv in self.population_dict[0].iterrows():
                
                d_num_lot = ini_indiv['num_lot']
                d_operation = ini_indiv['operation']

                run_idx = indiv.index[(indiv.num_lot == d_num_lot) & (indiv.operation == d_operation)].tolist()[0]

                new_col.append(run_idx)
            gene_idx_in_pop[indiv_idx] = new_col
            unique_value = []
        for gene_idx, gene in gene_idx_in_pop.iterrows():
            occurence = len(gene.unique()) - 6
            unique_value.append(occurence)
        gene_idx_in_pop['unique_value'] = unique_value
                
        return gene_idx_in_pop, unique_value