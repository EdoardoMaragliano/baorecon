import sys

repo_path = '/home/emaragliano/Work/Projects/Dottorato/baorecon'
sys.path.insert(0, repo_path)

import os
'''
os.environ["OMP_NUM_THREADS"] = '8'
os.environ["OPENBLAS_NUM_THREADS"] = '8'
os.environ["MKL_NUM_THREADS"] = '8'
os.environ["VECLIB_MAXIMUM_THREADS"] = '8'
os.environ["NUMEXPR_NUM_THREADS"] = '8'
'''


from zeldareco.pipeline import ReconstructionPipeline
from numba import set_num_threads, get_num_threads


nth = 24
set_num_threads(nth)
print(f'setting {get_num_threads()} threads')

pipeline = ReconstructionPipeline(config_file='bao_pipeline_example.yaml')
data_path, random_path = pipeline.run()

print(data_path)
print(random_path)