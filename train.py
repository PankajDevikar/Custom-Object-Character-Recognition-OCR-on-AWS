import os
import argparse
import time
import torch
import torch.optim as optim
from models import Darknet  # This imports the updated Darknet

def train(opt):
    # Initialize YOLOv3 model using the provided model config and image size
    model = Darknet(opt.cfg, img_size=opt.img_size)
    
    # Load pre-trained weights if provided and valid (.pt file)
    if opt.weights and os.path.exists(opt.weights):
        if opt.weights.endswith(".pt"):
            print(f"Loading weights from {opt.weights}")
            try:
                model.load_state_dict(torch.load(opt.weights, map_location=torch.device("cpu")))
            except Exception as e:
                print(f"Error loading weights: {e}. Training from scratch.")
        else:
            print("The provided weights file does not have a '.pt' extension. "
                  "Please convert Darknet weights (.weights) to a PyTorch checkpoint (.pt) before training.")
            return
    else:
        print("No pre-trained weights found. Training from scratch.")
    
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    # Define optimizer (using SGD)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    # Create dummy training data (replace with your actual DataLoader)
    dummy_input = torch.randn(opt.batch_size, 3, opt.img_size, opt.img_size, device=device)
    # Dummy targets (not used in dummy loss but provided for compatibility)
    dummy_targets = torch.zeros((opt.batch_size, 10), device=device)
    
    # Training loop simulation: simulate 5 batches per epoch
    for epoch in range(opt.epochs):
        epoch_loss = 0.0
        for batch in range(5):
            optimizer.zero_grad()
            loss, _ = model(dummy_input, dummy_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {batch+1}, Loss: {loss.item():.4f}")
            time.sleep(0.1)  # simulate delay
        avg_loss = epoch_loss / 5.0
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Save the trained model weights
    save_path = os.path.join(os.getcwd(), "yolov3_trained.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset config file")
    parser.add_argument("--cfg", type=str, default="yolov3.cfg", help="Path to model config file")
    parser.add_argument("--weights", type=str, default="yolov3.pt", help="Path to pre-trained weights (.pt)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=416, help="Input image size for training")
    
    opt = parser.parse_args()
    print(opt)
    train(opt)
