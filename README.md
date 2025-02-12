
# MNIST Hydra Project

This project demonstrates how to use Hydra for configuration management in a PyTorch-based machine learning project. It trains a simple neural network on the MNIST dataset and uses GitHub Actions for CI/CD to run the training script and post results to pull requests.

## Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the training script: `python main.py`.

## GitHub Actions

The GitHub Actions workflow runs the training script on push and pull request events and posts the accuracy and model parameters as comments on pull requests.
