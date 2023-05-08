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

to_dock = pd.read_csv('input/to_vina_230508.csv')
box_root = 'input/klifs_boxes'

# THREAD NUMBER MUST BE DIVISIBLE BY 32!
threads = 8192
search_depth = 10

print('-'*50)
print(f'Threads: {threads}\nSearch_depth: {search_depth}')
print('-'*50)
sub_folder = f'{threads}_{search_depth}'

for pdb in to_dock['Structure ID'].unique():
    print("Currently working on target:", pdb)

    # Get target PDB path and output subfolder
    target_pdb_path = os.path.join('input', 'pdbs', str(pdb)+'.pdb')
    output_subfolder = '_'.join([str(pdb), sub_folder])

    # Get box coordinates
    if os.path.exists(os.path.join(box_root, str(pdb)+'_box.csv')):
        box_data = pd.read_csv(os.path.join(box_root, str(pdb)+'_box.csv'))
        box_center = box_data['center'].tolist()
        box_size = box_data['size'].tolist()
    else:
        print(f'No box data for {pdb} found. Using default box coordinates.')
        box_center = (1., 21.8, 36.3) # Active site coordinates 
        box_size = (30,30,30)

    # SKIP EXISTING RUNS
    if os.path.exists(f'output/{output_subfolder}/log.tsv'):
        continue

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
        search_depth=search_depth,
        threads=threads, 
        threads_per_call=threads,
        verbose=False,
        gpu_ids=[0],
        workers_per_gpu=1,
        num_cpu_workers=0)
