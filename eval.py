
import json
import os
import gc
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import time
from src.utils import setup_seed
from src.pixel_classifier import pixel_classifier
from src.feature_extractors import create_feature_extractor, collect_features
from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from transformers import AutoTokenizer, AutoModel
from flash_attn import flash_attn_qkvpacked_func
import pandas as pd
from PIL import Image
from torch.cuda.amp import autocast  # Import autocast
from utils import read_text
from src.load_dataset2 import RandomGenerator, ValGenerator, Mixdataset
import shutil
import numpy as np
import cv2
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_image_with_dice(predict_save, labs, save_path):
    
     # Ensure inputs are numpy, handle potential tensor inputs
    if isinstance(predict_save, torch.Tensor):
        predict_save = predict_save.detach().cpu().numpy()
    if isinstance(labs, torch.Tensor):
        labs = labs.detach().cpu().numpy()
        
        
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    
    # Ensure both arrays have the same shape (should be 256x256 now)
    target_shape = (256, 256)
    if tmp_lbl.shape != target_shape:
         print(f"Warning: Ground truth label shape is {tmp_lbl.shape}, resizing to {target_shape}.")
         tmp_lbl = cv2.resize(tmp_lbl, target_shape, interpolation=cv2.INTER_NEAREST)
    if tmp_3dunet.shape != target_shape:
        print(f"Warning: Prediction shape is {tmp_3dunet.shape}, resizing to {target_shape}.")
        tmp_3dunet = cv2.resize(tmp_3dunet, target_shape, interpolation=cv2.INTER_NEAREST)

    # Flatten for metric calculation
    tmp_lbl_flat = tmp_lbl.flatten()
    tmp_3dunet_flat = tmp_3dunet.flatten()

    dice_pred = 2 * np.sum(tmp_lbl_flat * tmp_3dunet_flat) / (np.sum(tmp_lbl_flat) + np.sum(tmp_3dunet_flat) + 1e-5)
    #iou_pred = jaccard_score(tmp_lbl, tmp_3dunet)
    # --- Calculate IoU (Jaccard) ---
    # Threshold the prediction for IoU (use the flattened array)
    # Convert both flattened arrays to boolean or integer 0/1 for jaccard_score with average='binary'
    intersection = np.sum((tmp_lbl_flat > 0.5) & (tmp_3dunet_flat > 0.5))
    union = np.sum((tmp_lbl_flat > 0.5) | (tmp_3dunet_flat > 0.5))
    iou_pred = (intersection + 1e-5) / (union + 1e-5)
    
    fig, ax = plt.subplots()
    ax.imshow(predict_save * 255, cmap='gray')
    
    rect = patches.Rectangle((58, 95), width=68, height=63, edgecolor='red', linewidth=1, facecolor='none')

    ax.add_patch(rect)
    #plt.text(x=10, y=24, s=f"Dice: {dice_pred:.3f}", fontsize=5, color='white')
    
    # Add Dice score text to the plot
    plt.text(x=5, y=15, s=f"Dice: {dice_pred:.3f}\nIoU: {iou_pred:.3f}", fontsize=8, color='white',
             bbox=dict(facecolor='black', alpha=0.5))
    plt.axis("off")
    plt.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return dice_pred, iou_pred

# Remove the old vis_and_save_heatmap function

def save_prediction_visualization(predict_save, labs, vis_save_path):
    """
    Saves a visualization of the prediction mask using show_image_with_dice.

    Args:
        predict_save (np.ndarray): The prediction mask (should be 2D).
        labs (np.ndarray): The ground truth label mask (should be 2D).
        vis_save_path (str): The path to save the visualization image.

    Returns:
        tuple: (dice_score, iou_score) calculated by show_image_with_dice, or (None, None) on error.
    """
    try:
        # Ensure inputs are NumPy arrays and have compatible shapes/types for show_image_with_dice
        if isinstance(predict_save, torch.Tensor):
            predict_save = predict_save.detach().cpu().numpy()
        if isinstance(labs, torch.Tensor):
            labs = labs.detach().cpu().numpy()

        # Ensure predict_save is 2D (remove potential batch/channel dims)
        if predict_save.ndim > 2:
            predict_save = predict_save.squeeze()
            if predict_save.ndim != 2:
                 raise ValueError(f"Prediction has unexpected dimensions after squeeze: {predict_save.shape}")

        # Ensure labs is 2D
        if labs.ndim > 2:
            labs = labs.squeeze()
            if labs.ndim != 2:
                 raise ValueError(f"Label has unexpected dimensions after squeeze: {labs.shape}")

        # Ensure predict_save is float32 as expected by show_image_with_dice
        # (show_image_with_dice does binary thresholding internally if needed)
        predict_save = predict_save.astype(np.float32)
        labs = labs.astype(np.float32) # Ensure labs is also float

        print(f"Saving visualization: Prediction shape {predict_save.shape}, Label shape {labs.shape}")

        # Call the visualization function
        dice_score, iou_score = show_image_with_dice(predict_save, labs, save_path=vis_save_path)

        # Optional: Add text like Dice score to the image if show_image_with_dice doesn't do it
        # img = cv2.imread(vis_save_path)
        # cv2.putText(img, f"Dice: {dice_score:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # cv2.imwrite(vis_save_path, img)

        return dice_score, iou_score

    except Exception as e:
        print(f"Error in save_prediction_visualization: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None, None

class CustomDataset(Dataset):
    def __init__(self, text_file, img_folder, label_folder, transform=None):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.text_data = pd.read_excel(text_file)
        self.transform = transform if transform else transforms.ToTensor()  # Use ToTensor() if no transform is provided

        # Create a case-insensitive mapping of label filenames
        self.label_files = {f.lower(): f for f in os.listdir(self.label_folder)}

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        img_name = self.text_data.iloc[idx]['Image'].strip()

        # Convert .png to .tif for image folder lookup
        img_name_tif = img_name.replace('.png', '.tif')

        # Load the image
        img_path = os.path.join(self.img_folder, img_name_tif)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            return None  # Skip this sample

        image = Image.open(img_path).convert('RGB')

        # Load the label
        label_name = self.label_files.get(img_name.lower(), None)  
        if label_name is None:
            print(f"Warning: Label not found for {img_name}")
            return None  # Skip this sample

        label_path = os.path.join(self.label_folder, label_name)
        label = Image.open(label_path).convert('L')  # Convert to grayscale for segmentation

        # Convert images to tensors
        image = self.transform(image)  # Convert image to tensor
        label = self.transform(label)  # Convert label to tensor

        # Get text description
        text_row = self.text_data.loc[self.text_data['Image'].str.lower() == img_name.lower()]
        text = text_row['Description'].values[0] if not text_row.empty else ""

        return {'image': image, 'label': label, 'name': img_name, 'text': text}



def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def logger_config(log_path):
    logger = logging.getLogger()
    logger.propagate = False
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

# Load tokenizer and model as done in load_dataset.py
tokenizer = AutoTokenizer.from_pretrained('siddharthdhara17/ClinicalModernBERT-4epochs', use_auth_token=True)
bert_embedding = AutoModel.from_pretrained('siddharthdhara17/ClinicalModernBERT-4epochs', use_auth_token=True)

def enable_flash_attention(model):
    model.to("cuda")
    for layer in model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.forward = lambda x: flash_attn_qkvpacked_func(x.to("cuda"), causal=False)

enable_flash_attention(bert_embedding)  # Apply Flash Attention

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return loss

    def forward(self, preds, target, weight=None, softmax=False):
        if softmax:
            preds = torch.softmax(preds, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert preds.size() == target.size(), 'predict {} & target {} shape do not match'.format(preds.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(preds[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice_loss.item())
            loss += dice_loss * weight[i]
        return loss / self.n_classes


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.coef_bce =  1.0
        self.coef_dice = 1.5

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        diceloss = 1 - dice.sum() / num
        return self.coef_bce * bce + self.coef_dice * diceloss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.device().numpy()
    if torch.is_tensor(target):
        target = target.data.device().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou+1)
    return iou, dice


def evaluation(args, model, extractor, valloader):
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=device).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], generator=rnd_gen, device=device)
    else:
        noise = None
    preds, gts, uncertainty_scores = [], [], []
    for idx, sample in enumerate(valloader):
        img, label, text, name = sample['image'], sample['label'], sample['text'], sample['name']
        img = img.to(device)
        text = text.to(device)
        features = extractor(img, noise=noise)
        features = collect_features(activations=features)

        for k, v in features.items():
            features[k] = features[k].to(text.device)
        with torch.no_grad():
            pred = model(features, text)
            assert pred.dim() == 4 and pred.shape[1] > 1, "pred outputs should have >1 classes"
            pred_softmax = torch.softmax(pred, dim=1)
            _, pred = torch.max(pred_softmax, dim=1)
            pred = pred.cpu().numpy()  # Move tensor to CPU before converting to NumPy
            labs = label[0].device().numpy()
            predict_save = pred[0]
            predict_save = np.reshape(predict_save, (args['image_size'], args['image_size']))
            current_directory = os.getcwd()
            save_path = os.path.join(current_directory, "_boundingbox")
            dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=save_path + '.jpg')

        gts.append(label.numpy())
        preds.append(pred)  # pred is already a NumPy array

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    for pred, target in zip(preds, gts):
        iou, dice = iou_score(pred, target)
        iou_avg_meter.update(iou, target.shape[0])
        dice_avg_meter.update(dice, target.shape[0])

    return dice_avg_meter.avg, iou_avg_meter.avg

def test(args, model, extractor, test_loader):
    model.to(device)  # Ensure model is on GPU
    extractor.to(device)  # Ensure feature extractor is on GPU
    model.eval()
    preds, gts = [], []

    save_path = os.path.join(os.getcwd(), "_test_boundingbox")
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

    total_images = len(test_loader)
    print(f"Total images in test set: {total_images}")

    for idx, sample in enumerate(test_loader):
        print(f"Processing {idx+1}/{total_images}...")  # Track progress

        img, label, text, name = sample['image'], sample['label'], sample['text'], sample['name']
        
        # Move data to GPU
        img, label = img.to(device), label.to(device)

        if isinstance(text, list):  
            text = text[0]  # Ensure text is properly formatted

        # Extract features on GPU
        with torch.no_grad():
            features = extractor(img)
            features = collect_features(activations=features)

            # Ensure all features are moved to GPU
            for k, v in features.items():
                features[k] = v.to(device)

            # Perform inference with mixed precision
            with autocast(device_type="cuda", dtype=torch.float16):
                pred = model(features)

            assert pred.dim() == 4 and pred.shape[1] > 1, "Predictions should have >1 classes"
            pred_softmax = torch.softmax(pred, dim=1)
            _, pred = torch.max(pred_softmax, dim=1)
            pred = pred.device().numpy()  

        predict_save = pred[0]
        predict_save = np.reshape(predict_save, (img.shape[-2], img.shape[-1]))

        # Save bounding box results
        bbox_save_path = os.path.join(save_path, f"{name[0]}.jpg")
        try:
            dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, label.device().numpy(), save_path=bbox_save_path)
            print(f"Bounding box image saved to: {bbox_save_path}")
        except Exception as e:
            print(f"Error saving bounding box image for {name[0]}: {e}")

        # Save heatmap results
        vis_save_path = os.path.join(save_path, f"{name[0]}_heatmap.jpg")
        try:
            vis_and_save_heatmap(model, img, text, img_RGB=None, labs=label, vis_save_path=vis_save_path)
            print(f"Heatmap image saved to: {vis_save_path}")
        except Exception as e:
            print(f"Error saving heatmap image for {name[0]}: {e}")

        gts.append(label.device().numpy())  # Move back to CPU for logging
        preds.append(pred)

    print("Test evaluation complete. Heatmaps and bounding box images saved to:", save_path)


# Image transform: Resize to 256, convert to tensor, normalize to [-1, 1]
image_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR), # Resize PIL Image
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Label transform: Resize to 256 (nearest), convert to tensor
# We will resize the numpy array later using cv2.INTER_NEAREST for better control
label_transform_to_tensor = transforms.ToTensor()


def infer_single_image(model, extractor, img_path, label_path, text_file, tokenizer, bert_embedding, device, target_size=256):
    """
    Runs inference on a single image using the TextDiff model components.
    Args:
        model: The pixel_classifier model.
        extractor: The feature extractor model.
        img_path (str): Path to the input image.
        label_path (str): Path to the ground truth label image.
        text_file (str): Path to the Excel file with descriptions.
        tokenizer: BERT tokenizer.
        bert_embedding: BERT model for embeddings.
        device: CUDA or CPU device.
        target_size (int): The target image size (should match training, e.g., 256).
    """
    model.to(device)
    extractor.to(device)
    model.eval()
    extractor.eval()

    #transform = transforms.ToTensor()  # Define necessary transformations
    #transform = transforms.Compose([
    #transforms.ToTensor(),  # Scales to [0, 1]
   #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Shifts to [-1, 1]
#])
    image = None

    # Load image
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
    try:
        img_pil = Image.open(img_path).convert('RGB')
        # Apply the correct image transform (resizes to target_size, normalizes)
        image = image_transform(img_pil).unsqueeze(0).to(device)
        print(f"Input image loaded and resized to: {image.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        import traceback
        traceback.print_exc()
        return

    if image is None:
        print("Error: Image processing failed.")
        return

    # --- Label Loading ---
    label_resized_np = None
    if not os.path.exists(label_path):
        print(f"Warning: Label not found at {label_path}. Proceeding without label.")
    else:
        try:
            label_pil = Image.open(label_path).convert('L') # Load as grayscale PIL
            # Convert PIL label to numpy array (H, W)
            label_np = np.array(label_pil)

            # Ensure label is binary (0 or 255) -> (0 or 1) for resizing
            if np.max(label_np) > 1:
                label_np = (label_np > 127).astype(np.uint8) # Binarize if needed

            print(f"Label shape numpy original: {label_np.shape}")

             # Resize numpy array using cv2 (INTER_NEAREST is crucial for masks)
            label_resized_np = cv2.resize(label_np, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            label_resized_np = label_resized_np.astype(np.float32) # Ensure float for dice calculation
            print(f"Label shape after resize: {label_resized_np.shape}")
        except Exception as e:
            print(f"Error loading/resizing label: {e}")
            import traceback
            traceback.print_exc()
            # Decide how to handle: return, or proceed without label
            return # Stop if label loading/processing fails


    # Extract text description from Test_text.xlsx
    # --- Text Processing ---
    text_input = "No description available." # Default text
    try:
        text_data = pd.read_excel(text_file)
        img_filename = os.path.basename(img_path)
        # Assign base_name HERE
        base_name = os.path.splitext(img_filename)[0]
        # Use base_name consistently
        matched_rows = text_data[text_data['Image'].str.contains(base_name, case=False, na=False)]
        if not matched_rows.empty:
             text_input = matched_rows['Description'].values[0]
             # Now printing base_name is safe
             print(f"Found description for {base_name}: '{text_input[:50]}...'")
        else:
             # Use base_name here too
             print(f"Warning: No description found for image filename containing '{base_name}'")
    except Exception as e:
        print(f"Error loading text description from {text_file}: {e}")
        
        # Continue with default text

    # **Generate text embedding using ClinicalModernBERT**
    textf = None
    try:
        print("Generating text embedding...")
        text_tokens = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512) # Added max_length
        with torch.no_grad(): # Ensure no gradients for embedding generation
             textf = bert_embedding(**text_tokens.to(device)).last_hidden_state  # Keep full sequence output
        print(f"Text embedding generated with shape: {textf.shape}") # Verify shape
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        import traceback
        traceback.print_exc()
        return # Cannot proceed without text embedding

    # --- Feature Extraction & Inference ---
    predict_save = None
    print("Extracting features and running inference...")
    try:
        with torch.no_grad():
            features = extractor(image) # image is already [1, 3, target_size, target_size]
            features = collect_features(activations=features)

            for k, v in features.items():
                features[k] = v.to(device)

            # Use autocast if desired and beneficial
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                 # Pass features AND text embedding to the classifier
                 pred = model(features, textf=textf)

            # --- Output Processing ---
            # Ensure prediction is float32 for softmax
            pred = pred.float()
            print(f"Raw prediction shape: {pred.shape}") # Should be [1, num_classes, target_size, target_size]

            if pred.shape[2:] != (target_size, target_size):
                 print(f"Warning: Model output shape {pred.shape[2:]} doesn't match target {target_size}. Check model architecture.")
                 # Optional: Resize output if necessary, but indicates a potential model definition issue
                 pred = F.interpolate(pred, size=(target_size, target_size), mode='bilinear', align_corners=False)


            assert pred.dim() == 4 and pred.shape[1] > 1, "Prediction output should have >1 classes"

            pred_softmax = torch.softmax(pred, dim=1)
            # Assuming class 1 is the foreground mask
            # pred_mask = pred_softmax[:, 1, :, :] # Get probability map for class 1
            # _, pred_class_idx = torch.max(pred_softmax, dim=1) # Get class indices
            pred_class_idx = torch.argmax(pred_softmax, dim=1) # Use argmax is clearer

            # Convert predicted class indices to a float mask (0.0 or 1.0)
            pred_mask_np = pred_class_idx.squeeze(0).cpu().numpy().astype(np.float32) # Remove batch dim, move to CPU, convert

        predict_save = pred_mask_np # Shape [target_size, target_size]
        print(f"Final prediction mask shape: {predict_save.shape}")

    except Exception as e:
        print(f"Error during feature extraction or inference: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Save results ---
    img_basename = os.path.splitext(os.path.basename(img_path))[0] # Get filename without extension
    save_dir = os.path.join(os.getcwd(), "single_image_result_256")
    os.makedirs(save_dir, exist_ok=True)
    #bbox_save_path = os.path.join(save_path, f"{img_basename}_pred.png") # Save prediction as png
    # vis_save_path = os.path.join(save_path, f"{img_basename}_vis.jpg") # Path for visualization with scores
    pred_save_path = os.path.join(save_dir, f"{img_basename}_pred.png")
    # Save the raw prediction mask
    try:
        # Convert float mask (0.0, 1.0) to uint8 (0, 255) for saving as image
        predict_save_img = (predict_save * 255).astype(np.uint8)
        #v2.imwrite(bbox_save_path, predict_save_img)
        cv2.imwrite(pred_save_path, predict_save_img)
        print(f"Prediction mask saved to: {pred_save_path}")
        #print(f"Prediction mask saved to: {bbox_save_path}")
    except Exception as e:
        print(f"Error saving prediction mask image: {e}")


    # Create and save visualization using show_image_with_dice if label exists
    if label_resized_np is not None and predict_save is not None:
        vis_save_path = os.path.join(save_dir, f"{img_basename}_vis.jpg")
        print("Attempting to save visualization with Dice/IoU...")
        try:
            # show_image_with_dice expects float inputs
            dice_score, iou_score = show_image_with_dice(predict_save, label_resized_np, save_path=vis_save_path)
            if dice_score is not None:
                 print(f"Visualization with scores saved to: {vis_save_path} (Dice: {dice_score:.4f}, IoU: {iou_score:.4f})")
            else:
                 print(f"Visualization saved to: {vis_save_path}, but score calculation failed.")
        except Exception as e:
            print(f"Error saving visualization image using show_image_with_dice: {e}")
            import traceback
            traceback.print_exc()
    elif predict_save is not None:
         print("Label not available, skipping visualization with Dice/IoU scores.")
    else:
         print("Prediction failed, skipping saving results.")


    print(f"Inference complete for {img_path}.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    parser.add_argument('--exp', type=str, help="Path to config json (used for model dims, etc.)")
    parser.add_argument('--seed', type=int,  default=40)
    parser.add_argument('--checkpoint', type=str, default="/workspace/FastTextDiff/saved_textdiff/monuseg_2/experiment-02/model_best.pth", help="Path to the trained model checkpoint")
    parser.add_argument('--img_path', type=str, default="/workspace/FastTextDiff/datasets/monuseg_2/Test_Folder/img/TCGA-AC-A2FO-01A-01-TS1.tif", help="Path to the single input image")
    parser.add_argument('--label_path', type=str, default="/workspace/FastTextDiff/datasets/monuseg_2/Test_Folder/labelcol/TCGA-AC-A2FO-01A-01-TS1.png", help="Path to the corresponding label image")
    parser.add_argument('--text_file', type=str, default="/workspace/FastTextDiff/datasets/monuseg_2/Test_Folder/Test_text.xlsx", help="Path to the Excel file with text descriptions")


    args = parser.parse_args()
    setup_seed(args.seed)

    # Load necessary opts from the experiment config if provided
    opts = {}
    
    # !! IMPORTANT: Double-check if dim=[512, 512, 256, 256] is truly correct for the model arch !!
    default_opts = {
        'image_size': 256,
        'dim': [512, 512, 256, 256], # From ddpm.json - VERIFY THIS!
        'steps': [50, 150, 250],      # From ddpm.json
        # Add any other essential args needed by create_feature_extractor or pixel_classifier
        # that are present in ddpm.json (e.g., num_classes, blocks, etc.)
        'number_class': 2, # From ddpm.json
        'blocks': [6, 8, 12, 16] # From ddpm.json
    }
    
    if args.exp and os.path.exists(args.exp):
        print(f"Loading options from: {args.exp}")
        try:
            with open(args.exp, 'r') as f:
                opts_from_json = json.load(f)
            print(f"Successfully loaded JSON: {args.exp}")
             # Start with defaults, override with JSON, then override with command line args
            opts = default_opts.copy()
            opts.update(opts_from_json)
            opts.update(vars(args)) # Command line args take highest precedence
        except Exception as e:
             print(f"Error loading JSON {args.exp}: {e}. Using defaults + command line args.")
             opts = default_opts.copy()
             opts.update(vars(args))
    else:
        print(f"Warning: Experiment config file '{args.exp}' not found or not provided.")
        print("Using default values combined with command line arguments.")
        opts = default_opts.copy()
        opts.update(vars(args))

    # --- CRITICAL: Verify Final Configuration ---
    print("\n--- Final Configuration (`opts`) ---")
    print(f"image_size: {opts.get('image_size')}")
    print(f"dim: {opts.get('dim')}")
    print(f"steps: {opts.get('steps')}")
    print(f"number_class: {opts.get('number_class')}")
    print(f"blocks: {opts.get('blocks')}") # Example if needed by extractor
    # Add other important opts here
    print("------------------------------------\n")

    # Ensure required opts are present
    required_keys = ['image_size', 'dim', 'steps', 'number_class']
    if not all(key in opts for key in required_keys):
         missing = [key for key in required_keys if key not in opts]
         print(f"Error: Missing critical options in configuration: {missing}")
         exit()

    # --- Feature Extractor Initialization ---
    print("Initializing feature extractor...")
    try:
        # Pass necessary opts from the final config dictionary
        fea_extractor = create_feature_extractor(**opts).to(device)
        fea_extractor.eval()
        print("Feature extractor initialized.")
    except Exception as e:
        print(f"Error initializing feature extractor: {e}. Check config options and model code.")
        import traceback
        traceback.print_exc()
        exit()

    # --- Classifier Model Initialization ---
    print("Initializing classifier model...")
    try:
        # Calculate expected input dimensions for the classifier based on FINAL opts
        classifier_dims = [v * len(opts['steps']) for v in opts['dim']]
        print(f"Calculated classifier input dims based on dim={opts['dim']} and steps={opts['steps']}: {classifier_dims}")

        # Pass the calculated dimensions and number of classes to the classifier
        # Make sure pixel_classifier accepts these arguments
        classifier = pixel_classifier(extract_dims=classifier_dims).to(device) # Pass only extract_dims
        print("Classifier initialized.")
    except Exception as e:
        print(f"Error initializing pixel_classifier: {e}. Check config options and model code.")
        import traceback
        traceback.print_exc()
        exit()

    # --- Load Classifier Checkpoint ---
    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading trained classifier checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Adjust key based on how checkpoint was saved
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                 state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint # Assume it's the state_dict directly

            # Optional: Filter/modify keys if needed (e.g., due to DataParallel)
            # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            classifier.load_state_dict(state_dict)
            classifier.eval()
            print("Classifier checkpoint loaded successfully.")
        except Exception as e:
            print(f"Error loading classifier checkpoint from {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            exit()
    else:
        print(f"Error: Classifier checkpoint file not found at {checkpoint_path}")
        exit()

    # --- Load Tokenizer and BERT ---
    print("Loading tokenizer and BERT model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('siddharthdhara17/ClinicalModernBERT-4epochs', use_auth_token=True)
        bert_embedding = AutoModel.from_pretrained('siddharthdhara17/ClinicalModernBERT-4epochs', use_auth_token=True).to(device)
        bert_embedding.eval()
        print("Tokenizer and BERT model loaded.")
        # enable_flash_attention(bert_embedding) # Optional
    except Exception as e:
        print(f"Error loading tokenizer or BERT model: {e}")
        import traceback
        traceback.print_exc()
        exit()


    # --- Run Inference ---
    torch.cuda.empty_cache()
    gc.collect()

    print("\nStarting single image inference with TARGET_SIZE = 256...")
    infer_single_image(
        model=classifier,
        extractor=fea_extractor,
        img_path=args.img_path,
        label_path=args.label_path,
        text_file=args.text_file,
        tokenizer=tokenizer,
        bert_embedding=bert_embedding,
        device=device,
        target_size=opts['image_size'] # Pass the final image size
    )

    print("\nProgram finished.")