# Piano: pulizia worktree + riallineamento di `feat/multigpu` su `dev`

## Context

`main` e `dev` sono ora allineati e pushati (entrambi a `f4477fb`), quindi **"aggiornare da dev" e "aggiornare da main" sono la stessa cosa**: il riferimento unico è `dev`.

Restano due cose in sospeso:

1. Il branch `claude/teogpu02-benchmark-results-204ec6` e il suo worktree non hanno più contenuto proprio (puntano a `f4477fb`, cioè al tip di `dev`/`main`): sono da rimuovere.
2. `feat/multigpu` è in **doppio disallineamento**: divergente dal proprio remoto (locale +1 / origin +2) e fermo al merge-base `3f104ef` del 10 luglio, quindi **27 commit dietro `dev`** — gli mancano tutto `v0.6.0` e `v0.7.0`: packaging PEP 621, refactor dell'API pubblica di displacement, coverage/CI, e l'ottimizzazione VRAM del solver GPU.

Il branch contiene ~3.300 righe di lavoro reale sul multi-GPU (slab decomposition, `DistributedFFT`, MAS distribuito con ghost zone, launcher single-process, 4 file di test, 3 documenti). Va rimesso in pari **ora**, prima che la distanza cresca ancora e prima che la risoluzione dei conflitti diventi archeologia.

**Obiettivo di questo piano:** riallineare e avere i test verdi. Il merge di `feat/multigpu` → `dev` (e quindi CHANGELOG, versioning, extra `multigpu` in `pyproject.toml`) è **fuori scope**, si deciderà dopo.

**Strategia scelta: merge di `dev` in `feat/multigpu`**, non rebase. `origin/feat/multigpu` è già pubblicato e contiene commit di un'altra sessione: il rebase riscriverebbe storia condivisa (force-push) e riapplicherebbe il conflitto su `gpu.py` a ogni commit replayato. Il merge lo risolve **una volta sola**.

---

## Step 0 — Rimuovere il worktree e il branch `claude/teogpu02-…` — ✅ FATTO

> Eseguito il 2026-08-31. `plot_august.py` del worktree era identico alla copia
> (untracked) nel worktree principale, che resta: nessuna perdita. Il branch
> puntava a `f4477fb`, antenato di `dev`, e non esisteva su origin.

Il worktree contiene un file **untracked** `benchmarks/plot_august.py`; `git worktree remove` rifiuterà senza `--force`. Prima di forzare, verificare che non sia lavoro da salvare (in `baorecon/` ce n'è uno omonimo, anch'esso untracked — potrebbero non essere identici):

```bash
cd /mnt/project_mnt/home_fs/emaragliano/Work/Projects/Dottorato/baorecon
diff .claude/worktrees/teogpu02-benchmark-results-204ec6/benchmarks/plot_august.py \
     benchmarks/plot_august.py
```

Se identico (o non serve):

```bash
git worktree remove --force .claude/worktrees/teogpu02-benchmark-results-204ec6
git branch -d claude/teogpu02-benchmark-results-204ec6   # -d basta: e' antenato di dev
```

Il branch **non esiste su origin**, quindi nessun push di cancellazione.

---

## Step 1 — Riunire locale e remoto di `feat/multigpu`

I tre commit divergenti toccano file diversi, non ci si aspettano conflitti:

- locale: `2445374` *pyfftw backend docu*
- origin: `cdce463` *perf: O(P) host allreduce*, `8e6eeeb` *fix: warn when BAOReconstructor runs under MPI without a dist env*

```bash
git checkout feat/multigpu
git pull --rebase origin feat/multigpu
git push origin feat/multigpu     # allineati: 12 commit dal merge-base
```

Da qui in poi locale e origin coincidono.

---

## Step 2 — Merge di `dev` e risoluzione dei 3 conflitti

```bash
git merge dev
```

Conflitti attesi (verificati con un dry-run `git merge-tree`, riproducibile):

| File | Tipo |
|---|---|
| `requirements/gpu.txt` | modify/delete |
| [baorecon/solvers/fft/gpu.py](baorecon/solvers/fft/gpu.py) | content — **il conflitto di merito** |
| [baorecon/reconstruction/bao_reconstructor.py](baorecon/reconstruction/bao_reconstructor.py) | content |

### 2a. `requirements/gpu.txt` — banale

`dev` ha **eliminato l'intera cartella `requirements/`** passando agli extra di `pyproject.toml`; `feat/multigpu` ci aveva aggiunto `mpi4py`.

```bash
git rm requirements/gpu.txt
```

`cupy-cuda12x` è già nell'extra `gpu` di `pyproject.toml`. `mpi4py` **non va perso**: aggiungerlo come nuovo extra accanto agli esistenti (`test`, `notebook`, `gpu`, `docs`):

```toml
multigpu = [
    "mpi4py>=3.1,<5",
]
```

Le note che erano nei commenti del file (NCCL arriva col wheel `cupy-cuda12x`, `mpi4py` richiede una MPI di sistema) vanno spostate in [docs/multigpu.md](docs/multigpu.md), non perse.

### 2b. `baorecon/solvers/fft/gpu.py` — il nodo vero

**Entrambi i lati hanno ottimizzato la VRAM dello stesso hot path, in modi diversi e parzialmente sovrapposti.**

- **`feat/multigpu`** (`9d7fd6f`, +160/−46): introduce l'`ElementwiseKernel` `_scale_component_k`, che fonde `-i k_a / (bias k²)` in un'unica passata calcolando `|k|²` al volo dalle tre 1-D. Elimina `inv_k2_bias` (~0.5 griglia) **e** `scaled_k` (~1 griglia). Cruciale: la gestione del DC è **posizionale** (`k² == 0 → 0`), ed è proprio questo che lo rende corretto su uno **slab di ky** senza casi speciali per rank.
- **`dev`** (`0896f73`, +52/−12): scala `delta_k` **in place**, ricicla il buffer dell'ultimo `grad_i` come `proj_scratch`, e aggiunge rilasci di lifetime (`correction = None` a inizio iterazione e prima dell'allocazione del displacement).

**Risoluzione: base = versione `feat/multigpu`, sopra cui riapplicare a mano le ottimizzazioni di lifetime di `dev`.** La versione multigpu è strettamente più generale (serve alla correttezza distribuita) e sussume già l'eliminazione di `scaled_k` fatta da `dev`; quello che `dev` aggiunge in più sono i lifetime, ortogonali al kernel fuso. Concretamente, partendo dal lato multigpu:

1. **Tenere** `_scale_component_k` e tutti i suoi call site (ramo `axis`, ramo `radial`, `_compute_displacement_mesh`, `_compute_potential`).
2. **Eliminare** l'allocazione `proj_scratch = xp.empty(...)` nel setup del ramo radiale; dopo il loop sulle 3 componenti, riciclare il buffer morto: `proj_scratch = grad_i; grad_i = None` (idea di `dev`).
3. **Aggiungere** `correction = None` in cima al corpo di ogni iterazione e dopo il loop, prima di allocare la griglia di displacement (3×).
4. **Aggiungere** `del proj_scratch` dopo `divergence_from_components`.
5. Aggiornare di conseguenza il `del s, proj_scratch` finale → `del s`.

**`read_displacement_at`: le due implementazioni sono incompatibili e va scelta quella di `dev`**, che è il contratto pubblico rilasciato in v0.7.0 e documentato in [baorecon/solvers/_interface.py](baorecon/solvers/_interface.py) (*"returning a host `(N, 3)` array"*):

| | `dev` (v0.7.0, da tenere) | `feat/multigpu` (da adattare) |
|---|---|---|
| `pbc` | argomento del **costruttore** (`self._pbc`) | argomento del **metodo**, default `True` |
| ritorno | numpy **host** | array **CuPy** |
| MAS | normalizzato con `format_mas(mas)` | passato grezzo |

Il metodo risultante mantiene la firma di `dev` — `read_displacement_at(self, positions, mas="CIC")` — e al suo interno apre il ramo distribuito di multigpu (`halo_exchange_copy` sul `_displacement_ext`, sync di stream, `read_field_at`), convertendo a host con `cp.asnumpy` prima del `return` per rispettare il contratto. `format_mas` esiste su entrambi i lati ([baorecon/utils/formatters.py:159](baorecon/utils/formatters.py#L159)), nessun problema.

Il `__init__` risultante prende **entrambi** i parametri nuovi: `pbc=False` (da `dev`) e `dist=None` (da multigpu).

`_common.py` non confligge: il `v_k = None` di `dev` in `divergence_from_components` arriva gratis.

### 2c. `baorecon/reconstruction/bao_reconstructor.py`

I due lati hanno toccato la stessa zona per motivi indipendenti:

- **`dev`**: ha reso pubblici `_interpolate_displacement` → `interpolate_displacement` e `_get_rsd_displacement` → `get_rsd_displacement` (con docstring), e ha **cancellato** il vecchio ramo `if self._solver_type == 'ifft': …` delegando tutto a `solver.read_displacement_at`. Passa `pbc=self._pbc` al solver.
- **`feat/multigpu`**: ha aggiunto il parametro `dist`, la guardia foot-gun MPI, l'import di `CUPY_AVAILABLE` da `baorecon.utils.backend`, e un blocco distribuito **prima** di quel ramo ifft.

**Risoluzione:** tenere integralmente il corpo di `dev` (nomi pubblici, docstring, delega a `read_displacement_at`) e re-inserire il blocco distribuito di multigpu subito dopo il `survey_to_box_frame`, dentro il metodo ora chiamato `interpolate_displacement`. Poiché il solver rispetta ora il contratto host-numpy (§2b), nel blocco distribuito **cade il `cp.asnumpy`**: `shifts[mask] = shifts_own` diretto.

Nella costruzione del solver ifft convivono i due kwargs:

```python
solver_kwargs = {"pbc": self._pbc}
if self._device == "gpu":
    solver_kwargs["dist"] = self._dist
```

(`pbc` è accettato da entrambi i solver FFT dopo v0.7.0; `dist` solo dal GPU.)

Aggiornare anche i call site rinominati che multigpu non conosce (`_shift_gals`, `_shift_randoms` usano già i nomi pubblici sul lato `dev`: il merge li porta, va solo verificato che nessun residuo chiami `self._interpolate_displacement`).

### 2d. File auto-mergiati — da rileggere, non da dare per buoni

- [baorecon/pipeline/bao_pipeline.py](baorecon/pipeline/bao_pipeline.py): `dev` ha cambiato la firma dell'helper interno `_drop_device_grid(attr, …)` → `_drop_device_grid(obj, attr, …)` e aggiunto l'output `grid_density`; multigpu ha aggiunto `dist_env`, il gather degli slab in `_save_fits_image` e lo skip del pickle in modalità distribuita. **Verificare che dopo il merge non resti nessuna chiamata a `_drop_device_grid` con la vecchia aritmetica a 1 argomento** — git non può accorgersene.
- [baorecon/solvers/_interface.py](baorecon/solvers/_interface.py): auto-merge pulito (`dev` tocca la firma astratta, multigpu il blocco di import), ma controllare che la firma astratta e quella concreta di §2b coincidano.

```bash
grep -rn "_interpolate_displacement\|_get_rsd_displacement" baorecon/
grep -rn "_drop_device_grid" baorecon/pipeline/bao_pipeline.py
```

---

## Step 3 — Verifica

Usare il python dell'env **BAOFit** (`/usr/bin/python` non ha numpy):

```bash
PY=/farmdisk1/emaragliano/miniconda3/envs/BAOFit/bin/python
```

In ordine, fermandosi al primo rosso:

1. **Import e sanity serale** — il costo di sbagliare §2b è alto e si vede subito:
   ```bash
   $PY -c "import baorecon; from baorecon.solvers.fft.gpu import FFTSolverGPU; print(baorecon.__version__)"
   ```
2. **Suite serale completa** — deve restare verde: è il contratto che `dev` ha rilasciato.
   ```bash
   $PY -m pytest tests/ -x -q --ignore=tests/test_distributed.py --ignore=tests/test_distributed_gpu.py --ignore=tests/test_multidevice_gpu.py
   ```
3. **Equivalenza dei solver** — il test che cattura una risoluzione sbagliata del kernel fuso in `gpu.py`:
   ```bash
   $PY -m pytest tests/test_solver_equivalence.py tests/test_fft_solver.py -q
   ```
4. **API pubblica di displacement** (il refactor v0.7.0 che multigpu non aveva mai visto):
   ```bash
   $PY -m pytest tests/test_bao_reconstructor/ -q
   ```
5. **Suite distribuita** (specifica di questo branch):
   ```bash
   $PY -m pytest tests/test_distributed.py tests/test_field_ops.py -q
   $PY -m pytest tests/test_distributed_gpu.py tests/test_multidevice_gpu.py -q   # richiede >=2 GPU
   ```
6. **Controprova numerica end-to-end**: una ricostruzione GPU serale prima/dopo il merge sulla stessa mock deve dare displacement identici entro tolleranza — è l'unico modo di dimostrare che l'unione delle due ottimizzazioni VRAM non ha cambiato la fisica. Usare [benchmarks/bench_bao_reconstructor.py](benchmarks/bench_bao_reconstructor.py) e confrontare anche il picco di memoria: **deve essere ≤ quello di ciascuno dei due rami separati**, altrimenti una delle due ottimizzazioni è andata persa nella risoluzione.

---

## Step 4 — Salvare il piano sul branch e pushare — ✅ PARZIALE

> Questo documento è già committato su `feat/multigpu`. Il **push è bloccato**
> finché non si esegue lo Step 1 (il branch è divergente dal proprio remoto):
> il `git pull --rebase` dello Step 1 porterà con sé anche questo commit.

Il documento va versionato accanto agli altri documenti multi-GPU già presenti su questo branch (`docs/multigpu.md`, `docs/multigpu_migration_plan.md`, `docs/multigpu_migration_audit.md`):

```bash
git push origin feat/multigpu   # dopo lo Step 1
```

Il commit di merge dello Step 2 va fatto **prima**, con un messaggio che dichiari le scelte di risoluzione (kernel fuso di multigpu + lifetime di dev; contratto host-numpy di `read_displacement_at`), perché sono decisioni di merito che non si ricostruiscono dal diff.

---

## Fuori scope (esplicitamente)

- Merge di `feat/multigpu` → `dev` e apertura della PR.
- Voce CHANGELOG, bump di versione, extra `multigpu` come feature dichiarata pubblicamente (l'extra in §2a serve solo a non perdere `mpi4py`).
- Il branch `legacy`: storia **orfana** (nessun merge-base con `dev`), 54 commit del vecchio pacchetto `zeldareco/`. Non è mergiabile né rebasabile e resta un archivio. Se serve recuperarne una fix puntuale (es. `4d3311c`/`3da288a` sui dtype nel multigrid, `b0e0a51` sul bug RSD del solver), va fatto a mano con un `git show` mirato — verifica separata, non parte di questo piano.
