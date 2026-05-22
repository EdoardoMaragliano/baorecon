import sys

repo_path = '/home/emaragliano/Work/Projects/Dottorato/baorecon'
sys.path.insert(0, repo_path)

from zeldareco.pipeline import ReconstructionPipeline


pipeline = ReconstructionPipeline(config_file='bao_pipeline_example.yaml')
pipeline.run()

print(data_path)
print(random_path)