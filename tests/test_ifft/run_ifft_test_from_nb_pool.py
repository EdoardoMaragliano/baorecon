import itertools
import os
import papermill as pm
import pandas as pd
import yaml
import multiprocessing


# to be set to avoid overloading the system with too many threads per notebook, since
# each notebook can already use multiple threads for numpy/scipy operations.
# === GESTIONE RISORSE CPU ===
MAX_TOTAL_CORES = 32        # Sostituisci con il limite massimo di core che vuoi usare in totale
THREADS_PER_NOTEBOOK = 8    # Quanti thread può usare internamente ogni singolo notebook

# Imposta i thread a livello di sistema per le librerie C/C++ sottostanti
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_NOTEBOOK)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_NOTEBOOK)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_NOTEBOOK)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(THREADS_PER_NOTEBOOK)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_NOTEBOOK)

# Calcola quanti worker in parallelo possiamo avviare senza superare il limite
# Usiamo math.floor (tramite //) per arrotondare per difetto e non sforare mai
NUM_PROCESSI = max(1, MAX_TOTAL_CORES // THREADS_PER_NOTEBOOK)

print(f"=== ALLOCAZIONE RISORSE ===")
print(f"Core massimi consentiti : {MAX_TOTAL_CORES}")
print(f"Thread per Notebook     : {THREADS_PER_NOTEBOOK}")
print(f"Notebook in parallelo   : {NUM_PROCESSI}")
print(f"Core totali stimati     : {NUM_PROCESSI * THREADS_PER_NOTEBOOK}\n")

# --- 1. DEFINIZIONE DELLA FUNZIONE WORKER ---
# Questa funzione verrà eseguita in parallelo da diversi processi.
# --- 1. DEFINIZIONE DELLA FUNZIONE WORKER ---
def esegui_singola_run(args):
    idx, combo, keys, base_out_dir = args
    params = dict(zip(keys, combo))
    
    # PAD è condizionato da PBC
    params['PAD'] = 0.0 if params['PBC'] else float(params['PAD'])
    
    # Crea un nome descrittivo
    los_str = str(params['LOS'])
    run_name = f"run_{idx}_{params['RSD_SPACE']}_nmesh{params['NMESH']}_mas{params['MAS']}_pbc{params['PBC']}_pad{params['PAD']}_los{los_str}_smrad{params['SM_RAD']}"
    
    run_dir = os.path.join(base_out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    params['OUTDIR'] = run_dir
    output_notebook = os.path.join(run_dir, f"executed_notebook.ipynb")
    
    print(f"[RUN {idx}] Inizio esecuzione: {run_name}")
    
    try:
        pm.execute_notebook(
            'test_BAOfft_local_los.ipynb',
            output_notebook,
            parameters=params,
            progress_bar=False 
        )
        
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
                
            # ==========================================
            # CORREZIONE CRITERIO FISICO APPLICATA QUI
            # ==========================================
            if ks_mag_stat is not None:
                test_passed = (ks_mag_stat < 0.01)  # Accettiamo un errore massimo del 1%
            else:
                test_passed = False
            
            run_result = {
                'FOLDER_NAME': run_name,          
                'FOLDER_PATH': run_dir,           
                **params,                         
                'KS_mag_stat': ks_mag_stat, 
                'KS_mag_pval': ks_mag_pval, 
                'Diff_Delta_%': diff_delta,
                'Diff_Phi_%': diff_phi,
                'Test_Passed': test_passed,
                'Status': 'OK'
            }
        else:
            run_result = {'FOLDER_NAME': run_name, 'FOLDER_PATH': run_dir, **params, 'KS_mag_stat': None, 'KS_mag_pval': None, 'Diff_Delta_%': None, 'Diff_Phi_%': None, 'Test_Passed': False, 'Status': 'File YAML non trovato'}
            
    except pm.exceptions.PapermillExecutionError as e:
        print(f"[RUN {idx}] Errore durante l'esecuzione di {run_name}")
        run_result = {'FOLDER_NAME': run_name, 'FOLDER_PATH': run_dir, **params, 'KS_mag_stat': None, 'KS_mag_pval': None, 'Diff_Delta_%': None, 'Diff_Phi_%': None, 'Test_Passed': False, 'Status': 'Errore Esecuzione'}
        
    print(f"[RUN {idx}] Conclusa.")
    return run_result

    
# --- 2. BLOCCO PRINCIPALE ---
if __name__ == '__main__':

    # === IMPOSTAZIONE DRY RUN ===
    DRY_RUN = False
    
    # Definisci lo spazio dei parametri da esplorare
    rsd_space_list = ['real_space', 'redshift_space']
    nmesh_list = [256, 512] 
    mas_list = ['CIC', 'TSC']
    pbc_list = [True, False]
    pad_list = [100, 200, 300]
    los_list = [None, 'x', 'y', 'z']
    sm_rad_list = [15.0]

    keys = ['RSD_SPACE', 'NMESH', 'MAS', 'PBC', 'PAD', 'LOS', 'SM_RAD']
    
    # 1. Generiamo TUTTE le combinazioni grezze
    raw_combinations = list(itertools.product(
        rsd_space_list, nmesh_list, mas_list, pbc_list, pad_list, los_list, sm_rad_list,
    ))

    # 2. FILTRIAMO I DUPLICATI
    unique_combinations = set()
    combinations = []
    
    for combo in raw_combinations:
        # Estraiamo i valori (devono seguire l'ordine di 'keys')
        rsd, nmesh, mas, pbc, pad, los, sm_rad = combo
        
        # Forziamo PAD a 0.0 se PBC è True, altrimenti teniamo il pad originale
        pad_effettivo = 0.0 if pbc else float(pad)
        
        # Ricreiamo la tupla con il PAD corretto
        combo_corretta = (rsd, nmesh, mas, pbc, pad_effettivo, los, sm_rad)
        
        # Se questa combinazione non l'abbiamo ancora vista, la aggiungiamo!
        if combo_corretta not in unique_combinations:
            unique_combinations.add(combo_corretta)
            combinations.append(combo_corretta)

    print(f"\n=== PREPARAZIONE TEST ===")
    print(f"Combinazioni generate da itertools: {len(raw_combinations)}")
    print(f"Combinazioni REALI da eseguire (senza duplicati PBC/PAD): {len(combinations)}")

    # Selettore Dry Run
    if DRY_RUN:
        print("\n[DRY RUN ATTIVA] - Esecuzione interrotta. Nessun notebook è stato eseguito.")
    else:
        base_out_dir = "/home/emaragliano/Work/Projects/myfarm-disk/LE3-reconstruction/test_zelda_outputs_ifft"
        os.makedirs(base_out_dir, exist_ok=True)
        
        # Prepariamo gli argomenti da passare alla funzione worker usando la lista 'combinations' filtrata
        worker_args = [(idx, combo, keys, base_out_dir) for idx, combo in enumerate(combinations)]
        
        num_processi = NUM_PROCESSI
        print(f"Avvio del pool con {num_processi} processi paralleli...\n")

        # Avvio del Pool
        with multiprocessing.Pool(processes=num_processi) as pool:
            # pool.map esegue la funzione per ogni elemento in worker_args e aspetta che tutti finiscano
            results_table = pool.map(esegui_singola_run, worker_args)

        # Raccogli e salva la tabella finale
        df_results = pd.DataFrame(results_table)
        df_results.to_csv(os.path.join(base_out_dir, "summary_results_ifft_256_512.csv"), index=False)

        print("\n=== RIASSUNTO TEST ===")
        columns_to_show = ['RSD_SPACE', 'LOS', 'NMESH', 'MAS', 'PBC', 'PAD', 'SM_RAD', 'Diff_Delta_%', 'KS_mag_stat', 'Test_Passed']
        print(df_results[columns_to_show].fillna('N/A').to_string(index=False))