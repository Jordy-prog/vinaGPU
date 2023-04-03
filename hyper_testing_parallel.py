import os
import pandas as pd
from vinagpu import parallel_dock
import time

box_center = (1., 21.8, 36.3) # Active site coordinates 
box_size = (30,30,30)

to_dock = pd.read_csv('input/230317_KLIFS_Ligands.csv')

# test_pdb = to_dock['Structure ID'].unique()[0]
# test_smiles_df = to_dock[to_dock['Structure ID'] == test_pdb]
# test_smiles = test_smiles_df['SMILES'].tolist()

# THREAD NUMBER MUST BE DIVISIBLE BY 32!

threads = [256, 512, 1024, 2048, 4096, 8192]
search_depths = list(range(5, 31, 5))

for num_threads in threads:
    for depth in search_depths:
        '''SKIPPING 256 + 5 and 256 + 10 because they were already done on 1 GPU'''
        if num_threads == 256 and (depth == 5 or depth == 10):
            continue

        print('-'*50)
        print(f'Threads: {num_threads}\nSearch_depth: {depth}')
        print('-'*50)
        sub_folder = f'{num_threads}_{depth}'

        t0 = time.time()

        for pdb in to_dock['Structure ID'].unique():
            print("Currently working on target:", pdb)
            target_pdb_path = os.path.join('input', 'pdbs', str(pdb)+'.pdb')
            output_subfolder = '_'.join([str(pdb), sub_folder])

            smiles_df = to_dock[to_dock['Structure ID'] == pdb] 
            smiles = smiles_df['SMILES'].tolist()
            print("Ligands for this target:", len(smiles), end='\n\n')

            parallel_dock(
                target_pdb_path=target_pdb_path,
                smiles=smiles,
                output_subfolder=output_subfolder, 
                box_center=box_center,
                box_size=box_size,
                search_depth=depth,
                threads=num_threads, 
                threads_per_call=num_threads,
                verbose=False,
                gpu_ids=[0,1],
                workers_per_gpu=2,
                num_cpu_workers=0)


        t_spend = time.time() - t0

        with open(f'output/{num_threads}_{depth}_timing.txt', 'w') as f:
            f.write(str(t_spend))
