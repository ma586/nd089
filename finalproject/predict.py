import json

import numpy as np
import torch
from PIL import Image

import model_definition
import predict_cmd_parser


class Predictor:
    def __init__(self):{
        print("Start predicting")
    }


def predict_image(model, path_to_image, top_k, category_names, device):
    print("Start predict image")

    model.eval()

    image_tensor = process_image(path_to_image)

    # Add batch dimension and send to device
    image_tensor = image_tensor.to(device, dtype=torch.float32).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model.forward(image_tensor)
        ps = torch.exp(output)
        top_probs, top_indices = ps.topk(top_k, dim=1)

    # Move results to CPU for processing
    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()


    # Mapp index to class the resolve with predicted nr
    index_to_class = {v: k for k, v in model.class_to_idx.items()}


    top_classes = [index_to_class[i] for i in top_indices]
    top_flowers = [category_names[c] for c in top_classes]

    print()
    sorted_items = sorted(category_names.items(), key=lambda item: int(item[0]))

    for key, value in sorted_items:
        print(f'"{key}": "{value}"')

    print()
    print()
    print(f"Image to categorize: {path_to_image}")
    print("Top 5 Predictions:")
    for i in range(len(top_flowers)):
        print(f"#{i+1}: {top_flowers[i]} with probability {top_probs[i]*100:.2f}%")

    print("End predict image")

def main():
    parser = predict_cmd_parser.create_predict_parser()
    args = parser.parse_args()

    path_to_image = args.path_to_image
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    valid_devices = ['cpu', 'cuda', 'mps']

    if gpu not in valid_devices:
        print(f"Error: {gpu} is not a valid device.")
        print("Please choose one of the following devices:")
        print(valid_devices)
        return

    if gpu == 'cpu':
        device = torch.device("cpu")
    else:
        device:str = (
                torch.device("mps") if torch.backends.mps.is_available()
                else torch.device("cuda") if torch.cuda.is_available()
                else torch.device("cpu")
            )

    model = load_model(checkpoint, device)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    predict_image(model, path_to_image, top_k, cat_to_name, device)

def set_device(gpu: str):
    if gpu == 'mps':
        device = torch.device("mps")
    elif gpu == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def process_image(image_path: str = None):

    try:
        pil_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Image at {image_path} not found.")

    # Get image dimensions
    w, h = pil_image.size

    # 2. Resize the image (shortest side to 256, maintaining aspect ratio)
    if h > w:
        new_h = int(256 * h / w)
        new_w = 256
    else:
        new_w = int(256 * w / h)
        new_h = 256

    # Use a high-quality resampling filter for resizing
    resized_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 3. Crop the center 224x224 portion
    left = (new_w - 224) / 2
    top = (new_h - 224) / 2
    right = (new_w + 224) / 2
    bottom = (new_h + 224) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    # 4. Convert to NumPy array and then to float (0-1)
    np_image = np.array(cropped_image, dtype=np.float32) / 255.0

    # 5. Normalize with mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalized_image = (np_image - mean) / std

    # 6. Transpose dimensions to (color, height, width) for PyTorch
    transposed_image = normalized_image.transpose((2, 0, 1))

    # Convert the NumPy array to a PyTorch tensor
    return torch.from_numpy(transposed_image)


def load_model(checkpoint: str, device: str) -> model_definition.FlowerModel:
    print("Start loading model")
    checkpoint = torch.load(checkpoint, map_location=device)

    model = model_definition.FlowerModel(hidden_units=checkpoint['hidden_units'], arch=checkpoint['arch'], device=device, class_to_idx=checkpoint['class_to_idx'])

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=0) # Use dummy lr
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load index to class mapping
    model.class_to_idx = checkpoint['class_to_idx']

    # Check right device
    print(next(model.parameters()).device)
    model.to(device)

    model.eval()
    print("End loading model")

    return model

if __name__ == '__main__':
    main()

#   python predict.py "./flower_data/test/1/image_06743.jpg" "./checkpoint.pth" --top_k 5 --category_names "./cat_to_name.json" --gpu mps
#   python predict.py "./flower_data/test/12/image_03996.jpg" "./checkpoint.pth" --top_k 5 --category_names "./cat_to_name.json" --gpu mps
#   python predict.py "./flower_data/test/78/image_01830.jpg" "./checkpoint.pth" --top_k 5 --category_names "./cat_to_name.json" --gpu mps