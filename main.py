import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
import torch.nn as nn
from src.model import SimpleNN
from src.dataset import get_data_loaders
from src.train import train, test

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = get_data_loaders(cfg.data.batch_size, cfg.data.num_workers)

    # Initialize model, criterion, and optimizer
    model = SimpleNN(cfg.model.input_size, cfg.model.hidden_size, cfg.model.output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    # Training loop
    for epoch in range(1, cfg.model.num_epochs + 1):
        train(model, train_loader, criterion, optimizer, device)
        accuracy = test(model, test_loader, criterion, device)

    # Save accuracy and parameters
    with open("accuracy.txt", "w") as f:
        f.write(f"{accuracy:.2f}")

    with open("params.txt", "w") as f:
        f.write(f"Learning Rate: {cfg.model.learning_rate}, Hidden Size: {cfg.model.hidden_size}")

if __name__ == "__main__":
    main()
