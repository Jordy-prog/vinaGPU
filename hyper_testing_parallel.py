import os
import pandas as pd
from vinagpu import parallel_dock
import time

box_center = (1., 21.8, 36.3) # Active site coordinates 
box_size = (30,30,30)

to_dock = pd.read_csv('input/230406_KLIFS_Ligands_VINA.csv')

# THREAD NUMBER MUST BE DIVISIBLE BY 32!

threads = [256, 512, 1024, 2048, 4096, 8192]
search_depths = list(range(5, 31, 5))

for num_threads in threads:
    for depth in search_depths:
        print('-'*50)
        print(f'Threads: {num_threads}\nSearch_depth: {depth}')
        print('-'*50)
        sub_folder = f'{num_threads}_{depth}'

        for pdb in to_dock['Structure ID'].unique():
            print("Currently working on target:", pdb)

            t0 = time.time()

            target_pdb_path = os.path.join('input', 'pdbs', str(pdb)+'.pdb')
            output_subfolder = '_'.join([str(pdb), sub_folder])

            # SKIP EXISTING RUNS
            if os.path.exists(f'output/{output_subfolder}/log.tsv'):
                continue

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
                workers_per_gpu=1,
                num_cpu_workers=0)

            t_spend = time.time() - t0  # Measured time per klifs structure

            try:
                with open(f'output/{num_threads}_{depth}_timing.txt', 'r') as f:
                    prev_time = float(f.read().strip())
            except (FileNotFoundError, ValueError):
                prev_time = 0

            with open(f'output/{num_threads}_{depth}_timing.txt', 'w') as f:
                f.write(str(t_spend + prev_time))
