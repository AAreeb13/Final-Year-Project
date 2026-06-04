from src.evaluation.DataLoader import DataLoader


class EvaluationHarness:
    def __init__(self, system, data_loader, output_dir):
        self.system = system
        self.data_loader = data_loader
        self.output_dir = output_dir


    def run_system(self, datapoint_id, human_approval=True):
        datapoint = self.data_loader.load_datapoint(datapoint_id)
        problem_statement = datapoint.get("project_prompt", "")
        run_config = {
            # Add any specific configuration for running the system on this datapoint
        }
        result = self.system.run(
            problem_statement=problem_statement,
            run_config=run_config,
            human_approval=human_approval
        )
