import argparse

def create_predict_parser() -> argparse.ArgumentParser:
    predict_parser = argparse.ArgumentParser(description="Program to classify flowers.")
    predict_parser.add_argument("path_to_image", default='./flower_data/valid/1/image_06739.jpg', type=str, help="Path to the training data directory.")
    predict_parser.add_argument("checkpoint", default='./checkpoint.pth', type=str, help="Path to the training data directory.")
    predict_parser.add_argument("--top_k", default=5, type=int, help="Path where to save the checkpoint model.")
    predict_parser.add_argument("--category_names", default='./cat_to_name.json', type=str, help="Path where to save the checkpoint model.")
    predict_parser.add_argument("--gpu", type=str, help="Use cpu, mps or cuda", default='mps')
    return predict_parser
