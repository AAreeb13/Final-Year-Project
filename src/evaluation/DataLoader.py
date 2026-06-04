from pathlib import Path

from src.evaluation.helper import load_yaml_file


class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path).expanduser().resolve()
        self.data_point_paths = []
        
        # check name is Revolutionising-SWE-dataset
        if self.data_path.name != "Revolutionising_SWE_dataset" and self.data_path.name != "Revolutionising-SWE-dataset":
            print(self.data_path.name)
            raise ValueError(f"Data path must point to the 'Revolutionising-SWE-dataset' folder. Provided path: {self.data_path}")
        print(f"DataLoader initialized with data path: {self.data_path}")

        # check if it is a directory
        if not self.data_path.is_dir():
            raise ValueError(f"Data path must be a directory. Provided path: {self.data_path}")
        
        # check if it exists
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist. Provided path: {self.data_path}")
        
        # check if it contains directory called dataset
        if not (self.data_path / "dataset").is_dir():
            raise ValueError(f"Data path must contain a directory called 'dataset'. Provided path: {self.data_path}")

    def load_datapoint(self, datapoint_id):
        # Implement logic to load a specific datapoint based on the ID
        dataset_path = self.data_path / "dataset"
        
        # match datapoint_id to the folder name in dataset
        
        folders_in_dataset = [f for f in dataset_path.iterdir() if f.is_dir()]
        for f in folders_in_dataset:
            # load project_info.yaml and check if the id matches
            try:
                project_info = load_yaml_file(f / "project_info.yaml")
                print(project_info)
                if project_info.get("project_id") == datapoint_id:
                    self.data_point_paths.append(f)
                    return project_info
            except Exception as e:
                continue
        raise ValueError(f"Datapoint with ID '{datapoint_id}' not found in dataset.")
    
    def list_datapoints(self):
        dataset_path = self.data_path / "dataset"
        datapoints = []
        folders_in_dataset = [f for f in dataset_path.iterdir() if f.is_dir()]
        for f in folders_in_dataset:
            try:
                project_info = load_yaml_file(f / "project_info.yaml")
                datapoints.append({
                    "project_id": project_info.get("project_id"),
                    # "project_prompt": project_info.get("project_prompt"),
                    # "project_type": project_info.get("project_type"),
                    # "difficulty": project_info.get("difficulty")
                })
            except Exception as e:
                continue
        return datapoints
        
if __name__ == "__main__":
    print(DataLoader("~/Documents/Revolutionising_SWE_dataset").list_datapoints())