from environment import Environment
from agent import Agent
from dataset import Dataset
import time
import torch

if __name__ == "__main__":
    device = "cuda"
    dataset = Dataset(
        device=device,
        data_folder="data",
        transformed_data_folder="data_transformed",
    )
    dataset.transform_all_data()
    dataset.split_files_train_val()
    time.sleep(2)
    agent = Agent(device=device)
    environment = Environment(agent=agent, data=dataset, commission_for_train=0)
    environment.train_agent(batch_size=30_000)
    environment.val_agent()
    torch.save(agent.model, "model/test.pth")




