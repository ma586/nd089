import argparse

def create_train_parser():
    train_parser = argparse.ArgumentParser(description="A program to train a model to recognize flowers.")
    train_parser.add_argument("data_directory", type=str, help="Path to the training data directory.")
    train_parser.add_argument("--save_dir", default='.', type=str, help="Path where to save the checkpoint model. Put '.' for the current directory.")
    train_parser.add_argument("--arch", type=str, help="Architecture, example resnet101", default="resnet101")
    train_parser.add_argument("--learning_rate", type=float, help="Learning rate", default=0.001)
    train_parser.add_argument("--hidden_units", type=int, help="Hidden units", default=512)
    train_parser.add_argument("--epochs", type=int, help="Nr of epochs", default=1)
    train_parser.add_argument("--gpu", type=str, help="Use cpu, mps or cuda", default='mps')
    return train_parser
