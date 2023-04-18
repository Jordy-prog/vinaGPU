import os
import pandas as pd
from vinagpu import parallel_dock
import time


def preprocess_data(output_folder, smiles_list):
    '''
    return a list of smiles that are not yet in results
    '''
    # Remove every SMILES+KLIFS combination that is already in results
    if os.path.exists(f'output/{output_folder}/log.tsv'):
        data = pd.read_csv(f'output/{output_folder}/log.tsv', delimiter='\t')
        existing_smiles = data['smiles'].tolist()

        to_dock = set(smiles_list) - set(existing_smiles)

        return list(to_dock)

    return smiles_list

box_center = (1., 21.8, 36.3) # Active site coordinates 
box_size = (30,30,30)

to_dock = pd.read_csv('input/230406_KLIFS_Ligands_VINA.csv')

exhaustivenesses = [1, 8]
n_cpu = 16

for exhaustiveness in exhaustivenesses:
    print('-'*50)
    print(f'Exhaustiveness: {exhaustiveness}')
    print('-'*50)
    sub_folder = f'{exhaustiveness}_cpu'

    t0 = time.time()

    for pdb in to_dock['Structure ID'].unique():
        print("Currently working on target:", pdb)
        target_pdb_path = os.path.join('input', 'pdbs', str(pdb)+'.pdb')
        output_subfolder = '_'.join([str(pdb), sub_folder])

        smiles_df = to_dock[to_dock['Structure ID'] == pdb] 
        smiles = smiles_df['SMILES'].tolist()

        smiles = preprocess_data(output_subfolder, smiles)

        # If this target has already been fully docked, skip it
        if len(smiles) == 0:
            print('Already docked!')
            continue

        print("Ligands for this target:", len(smiles), end='\n\n')

        parallel_dock(
            target_pdb_path=target_pdb_path,
            smiles=smiles,
            output_subfolder=output_subfolder, 
            box_center=box_center,
            box_size=box_size,
            exhaustiveness=exhaustiveness,
            verbose=True,
            gpu_ids=[],
            num_cpu_workers=n_cpu)

        t_spend = time.time() - t0

        with open(f'output/{exhaustiveness}_timing_cpu_230411.txt', 'w') as f:
            f.write(str(t_spend))
