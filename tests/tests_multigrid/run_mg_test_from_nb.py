import itertools
import os
import papermill as pm
import pandas as pd
import yaml

'''
# to be set to avoid overloading the system with too many threads per notebook, since
# each notebook can already use multiple threads for numpy/scipy operations.

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
'''

# 1. Definisci lo spazio dei parametri da esplorare
bias_list = [1.7]
growth_rate_list = [0.88]
rsd_space_list = ['real_space', 'redshift_space']
nmesh_list = [256] 
sm_rad_list = [15.0]
mas_list = ['CIC', 'TSC']
pbc_list = [True, False]
los_list = [None, 'x', 'y', 'z']
pad_list = [100, 200, 300]              # Solo usato se PBC è False
DRY_RUN = True

# Raccogli le chiavi e crea tutte le combinazioni possibili (AGGIUNTO 'LOS')
#keys = ['BIAS', 'GROWTH_RATE', 'RSD_SPACE', 'NMESH', 'SM_RAD', 'MAS', 'PBC', 'LOS']
keys = ['RSD_SPACE', 'NMESH', 'MAS', 'PBC', 'PAD', 'LOS', 'SM_RAD']
combinations = list(itertools.product(
    rsd_space_list, nmesh_list, mas_list, pbc_list, pad_list, los_list, sm_rad_list,
))

print(f"\n=== PREPARAZIONE TEST ===")
print(f"Totale combinazioni (notebook da generare): {len(combinations)}")

# Selettore Dry Run
if DRY_RUN:
    print("\n[DRY RUN ATTIVA] - Esecuzione interrotta. Nessun notebook è stato eseguito.")
    print("Per lanciare i test, imposta DRY_RUN = False nello script.\n")
    
else:

    results_table = []
    base_out_dir = "/home/emaragliano/Work/Projects/myfarm-disk/LE3-reconstruction/test_zelda_outputs"

    # 2. Cicla su tutte le combinazioni
    for idx, combo in enumerate(combinations):
        print(f"\n=== Esecuzione combinazione {idx+1}/{len(combinations)} ===")
        params = dict(zip(keys, combo))
        
        # PAD è condizionato da PBC
        params['PAD'] = 0.0 if params['PBC'] else float(params['PAD'])
        
        # Crea un nome descrittivo e una cartella per l'output (AGGIUNTO LOS AL NOME)
        los_str = str(params['LOS']) # Converte None in "None" per il nome della cartella
        run_name = f"run_{idx}_{params['RSD_SPACE']}_nmesh{params['NMESH']}_mas{params['MAS']}_pbc{params['PBC']}_pad{params['PAD']}_los{los_str}_smrad{params['SM_RAD']}"
        
        run_dir = os.path.join(base_out_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        params['OUTDIR'] = run_dir
        output_notebook = os.path.join(run_dir, f"executed_notebook.ipynb")
        
        print(f"\n--- Esecuzione {run_name} ---")
        
        try:
            pm.execute_notebook(
                'test_BAOmultigrid_local_los.ipynb',
                output_notebook,
                parameters=params
            )
            
            # 3. Leggi i risultati dal file YAML
            yaml_file = os.path.join(run_dir, 'results.yaml')
            
            if os.path.exists(yaml_file):
                with open(yaml_file, 'r') as f:
                    res_data = yaml.safe_load(f) or {}
                    
                if params['RSD_SPACE'] == 'redshift_space':
                    ks_data = res_data.get('ks_psi_rsd', {})
                else:
                    ks_data = res_data.get('ks_psi_realspace', {})
                    
                diff_data = res_data.get('percent_diff', {})
                
                ks_mag_stat = ks_data.get('ks_magnitude', None)
                ks_mag_pval = ks_data.get('ks_magnitude_pvalue', None)
                diff_delta = diff_data.get('delta_exceeds_0.1', None)
                diff_phi = diff_data.get('phi_exceeds_0.1', None)
                    
                if ks_mag_stat is not None and ks_mag_pval is not None:
                    test_passed = (ks_mag_pval > 0.05) and (ks_mag_stat < 0.1)
                else:
                    test_passed = False
                
                run_result = {
                    **params, 
                    'KS_mag_stat': ks_mag_stat, 
                    'KS_mag_pval': ks_mag_pval, 
                    'Diff_Delta_%': diff_delta,
                    'Diff_Phi_%': diff_phi,
                    'Test_Passed': test_passed,
                    'Status': 'OK'
                }
            else:
                run_result = {**params, 'Test_Passed': False, 'Status': 'File YAML non trovato'}
                
            results_table.append(run_result)
            
        except pm.exceptions.PapermillExecutionError as e:
            print(f"Errore durante l'esecuzione di {run_name}")
            run_result = {**params, 'Test_Passed': False, 'Status': 'Errore Esecuzione'}
            results_table.append(run_result)

    # 4. Raccogli e salva la tabella finale
    df_results = pd.DataFrame(results_table)
    df_results.to_csv(os.path.join(base_out_dir, "summary_results.csv"), index=False)

    print("\n=== RIASSUNTO TEST ===")
    # AGGIUNTA LA COLONNA 'LOS' DA MOSTRARE NEL RECAP FINALE
    columns_to_show = ['RSD_SPACE', 'LOS', 'NMESH', 'MAS', 'PBC', 'PAD', 'SM_RAD', 'Diff_Delta_%', 'KS_mag_stat', 'Test_Passed']
    print(df_results[columns_to_show].fillna('N/A').to_string(index=False))