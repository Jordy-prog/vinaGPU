import os
import pandas as pd
from vinagpu import parallel_dock
import time

box_center = (1., 21.8, 36.3) # Active site coordinates 
box_size = (30,30,30)

to_dock = pd.read_csv('input/230317_KLIFS_Ligands.csv')

exhaustivenesses = [1, 8]
n_cpu = 16

for exhaustiveness in exhaustivenesses:
    print('-'*50)
    print(f'Exhaustiveness: {exhaustiveness}')
    print('-'*50)
    sub_folder = f'{exhaustiveness}_cpu'

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
            exhaustiveness=exhaustiveness,
            verbose=False,
            gpu_ids=[],
            num_cpu_workers=n_cpu)

        t_spend = time.time() - t0  # Measured time per klifs structure

        try:
            with open(f'output/{exhaustiveness}_timing_cpu.txt', 'r') as f:
                prev_time = float(f.read().strip())
        except (FileNotFoundError, ValueError):
            prev_time = 0

        with open(f'output/{exhaustiveness}_timing_cpu.txt', 'w') as f:
            f.write(str(t_spend + prev_time))
