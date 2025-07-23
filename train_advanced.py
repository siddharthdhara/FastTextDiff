
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
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    
    # Ensure both arrays have the same shape
    if tmp_lbl.shape != tmp_3dunet.shape:
        tmp_3dunet = cv2.resize(tmp_3dunet, (tmp_lbl.shape[1], tmp_lbl.shape[0]))

    # Flatten the arrays to ensure they have the same shape for element-wise operations
    tmp_lbl = tmp_lbl.flatten()
    tmp_3dunet = tmp_3dunet.flatten()

    # Ensure both arrays have the same length
    if tmp_lbl.shape != tmp_3dunet.shape:
        raise ValueError(f"Shape mismatch: tmp_lbl has shape {tmp_lbl.shape}, tmp_3dunet has shape {tmp_3dunet.shape}")

    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl, tmp_3dunet)
    
    fig, ax = plt.subplots()
    ax.imshow(predict_save * 255, cmap='gray')
    rect = patches.Rectangle((4, 4), width=120, height=20, edgecolor='red', linewidth=1, facecolor='none')
    ax.add_patch(rect)
    #plt.text(x=10, y=24, s=f"Dice: {dice_pred:.3f}", fontsize=5, color='white')
    plt.axis("off")
    plt.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, text, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    try:
        # Ensure input tensors are on the correct device
        input_img = input_img.to(next(model.parameters()).device)

        # Convert text to float and move to the correct device
        if isinstance(text, torch.Tensor):
            text = text.float().to(next(model.parameters()).device)
        elif isinstance(text, np.ndarray):
            text = torch.tensor(text, dtype=torch.float32).to(next(model.parameters()).device)
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")

        # Perform inference
        with torch.no_grad():
            output = model(input_img, text)
            pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))

        # Convert prediction to NumPy
        predict_save = pred_class[0].detach().cpu().numpy()  # Detach before converting
        predict_save = np.reshape(predict_save, (input_img.shape[-2], input_img.shape[-1]))

        # Convert labels to NumPy
        if isinstance(labs, torch.Tensor):
            labs = labs.detach().cpu().numpy()  # Detach before calling .numpy()

        # Ensure dice_pred and dice_ens are floats
        dice_pred = float(dice_pred)
        dice_ens = float(dice_ens) if dice_ens is not None else None

        # Save the heatmap image
        dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path + '_predict.jpg')

        return dice_pred_tmp, iou_tmp

    except Exception as e:
        print(f"Error in vis_and_save_heatmap: {e}")
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


def infer_single_image(model, extractor, img_path, label_path, text_file, tokenizer, bert_embedding, device):
    model.to(device)
    extractor.to(device)
    model.eval()

    transform = transforms.ToTensor()  # Define necessary transformations
    image = None

    # Load image
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512))  # Resize before converting to tensor
        image = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    if image is None:
        print("Error: Image processing failed.")
        return

    # Load label
    if not os.path.exists(label_path):
        print(f"Error: Label not found at {label_path}")
        return
    try:
        label = Image.open(label_path).convert('L')
        label = transform(label).unsqueeze(0).to(device)  # Convert label to tensor
        print(f"Label shape before resizing: {label.shape}")  # Debugging
    except Exception as e:
        print(f"Error loading label: {e}")
        return

    # Extract text description from Test_text.xlsx
    try:
        text_data = pd.read_excel(text_file)
        img_name = os.path.basename(img_path)  # Extract filename
        text_row = text_data.loc[text_data['Image'] == img_name]  # Find row with matching image name
        text_input = text_row['Description'].values[0] if not text_row.empty else "No description available."
    except Exception as e:
        print(f"Error loading text description: {e}")
        return

    # **Generate text embedding using ClinicalModernBERT**
    text_tokens = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
    textf = bert_embedding(**text_tokens.to(device)).last_hidden_state  # Keep full sequence output

    # Extract features
    with torch.no_grad():
        features = extractor(image)  # Extract features
        features = collect_features(activations=features)

        # Ensure features are moved to GPU
        for k, v in features.items():
            features[k] = v.to(device)

        # Perform inference on GPU
        with autocast(dtype=torch.float16):
            pred = model(features, textf=textf)

        assert pred.dim() == 4 and pred.shape[1] > 1, "Prediction output should have >1 classes"
        pred_softmax = torch.softmax(pred, dim=1)
        _, pred = torch.max(pred_softmax, dim=1)
        pred = pred.cpu().numpy()  # Move tensor to CPU before converting to NumPy

    predict_save = np.array(pred[0], dtype=np.float32)

    # Resize if needed
    if predict_save.shape != (512, 512):
        predict_save = cv2.resize(predict_save, (512, 512))

    print(f"predict_save shape after resize: {predict_save.shape}")

    # Save results
    save_path = os.path.join(os.getcwd(), "single_image_result")
    os.makedirs(save_path, exist_ok=True)
    bbox_save_path = os.path.join(save_path, "TCGA-AC-A2FO-01A-01-TS1_bbox.jpg")
    vis_save_path = os.path.join(save_path, "TCGA-AC-A2FO-01-TS1_heatmap.jpg")

    label_np = label.cpu().numpy().squeeze()  # Remove batch & channel dimensions
    print(f"Label shape after squeeze: {label_np.shape}")  # Debugging

    # Ensure label is 2D before resizing
    if len(label_np.shape) != 2:
        print(f"Error: Expected 2D array for label, got shape {label_np.shape}")
        return

    # Now resize safely
    label_resized = cv2.resize(label_np, (512, 512))
    print(f"Label shape after resize: {label_resized.shape}")  # Debugging

    label_resized = torch.tensor(label_resized, dtype=torch.float32, device=device)
    print(f"Label shape after resize (as Tensor): {label_resized.shape}")  # Debugging

    # Convert prediction to binary
    predict_save = np.where(predict_save > 0.5, 1, 0).astype(np.uint8)
    label_resized_np = label_resized.detach().cpu().numpy().astype(np.uint8)

    try:
        dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, label_resized_np, save_path=bbox_save_path)
        print(f"Bounding box image saved to: {bbox_save_path}")
    except Exception as e:
        print(f"Error saving bounding box image: {e}")
        dice_pred_tmp, iou_tmp = None, None  # Avoid undefined variable error

    try:
        textf_np = textf.detach().cpu().numpy() if isinstance(textf, torch.Tensor) else textf
        dice_pred_tmp = float(dice_pred_tmp) if dice_pred_tmp is not None else 0.0
        iou_tmp = float(iou_tmp) if iou_tmp is not None else 0.0

        # Ensure textf_np is passed as a tensor
        textf_np = torch.tensor(textf_np, dtype=torch.float32)

        dice_pred_tmp, iou_tmp = vis_and_save_heatmap(model, image, textf_np, None, label_resized_np, vis_save_path, dice_pred_tmp, iou_tmp)
        print(f"Heatmap image saved to: {vis_save_path}")

    except Exception as e:
        print(f"Error saving heatmap image: {e}")

    print("Inference complete for the single image.")


def main(args, extractor, data_loader):
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=device).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], generator=rnd_gen, device=device)
    else:
        noise = None

    gc.collect()
    extract_dims = [v * len(opts['steps']) for v in opts['dim']]
    classifier = pixel_classifier(extract_dims=extract_dims)

    classifier.init_weights()
    classifier = classifier.cuda()
    
    num_trainable_params = count_trainable_params(classifier)
    
    criterion_cross_entro = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=2)
    # criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[60], gamma=0.1)

    stats = {'best_dice': 0., 'best_iou': 0., 'best_epoch': 0, 'best_ckpt': None}
    start_train_time = time.time()  # Track training start time

    for epoch in range(args['max_training']):
        classifier.train()
        epoch_start_time = time.time()
        for idx, sample in enumerate(data_loader):
            img, label, text, name = sample['image'], sample['label'], sample['text'], sample['name']
            img = img.to(device)
            label = label.to(device)
            text = text.to(device)
            features = extractor(img, noise=noise)
            features = collect_features(activations=features)
            for k, v in features.items():
                features[k] = features[k].to(device)
            y_pred = classifier(features, textf=text)
            y_batch = label.type(torch.long)
            optimizer.zero_grad()
            
            # loss = criterion_cross_entro(y_pred, y_batch)
            # loss += 1.5 * criterion_dice(y_pred, y_batch)
            
            # Compute BCE and Dice Loss separately
            bce_loss = criterion_cross_entro(y_pred, y_batch)
            dice_loss = criterion_dice(y_pred, y_batch)
            
            loss = bce_loss + 1.5 * dice_loss  # Total loss
            
            loss.backward()
            optimizer.step()
            
        epoch_time = time.time() - epoch_start_time  # Measure epoch time

        # Log losses separately

        scheduler.step()

        with torch.no_grad():
            eval_start_time = time.time()  # Track inference time
            eval_dice, eval_iou = evaluation(opts, classifier.eval(), extractor= extractor, valloader=val_loader)
            eval_time = time.time() - eval_start_time  # Calculate inference time
            
            if eval_dice > stats['best_dice']:
                stats['best_dice'] = eval_dice
                stats['best_iou'] = eval_iou
                stats['best_epoch'] = epoch
                stats['best_ckpt'] = classifier.state_dict()
                model_path = os.path.join(args['exp_dir'], 'model_' + f'{epoch:02d}.pth')
                torch.save({'model_state_dict': stats['best_ckpt']}, model_path)

            logger.info(f"Epoch {epoch:02d}: dice/iou= {eval_dice:.4f}/{eval_iou:.4f} ")

    saved_path = os.path.join(args['exp_dir'], 'model_best.pth')
    logger.info(f'Final model saved to: {saved_path} \n Best epoch: {stats["best_epoch"]}, Best Dice: {stats["best_dice"]:.4f}, Best IoU: {stats["best_iou"]:.4f}')
    shutil.copy(src=model_path, dst=saved_path)

    train_time = time.time() - start_train_time  # Measure total training time
    
    print("Starting evaluation on test set...")
    test(opts, classifier, extractor, test_loader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int,  default=40)

    args = parser.parse_args()
    setup_seed(args.seed)
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))

    os.makedirs(opts['exp_dir'], exist_ok=True)
    opts['exp_dir'] = os.path.join(opts['exp_dir'], f'experiment-{len(os.listdir(opts["exp_dir"]))+ 1:02d}')
    os.makedirs(opts['exp_dir'], exist_ok=True)
    print('Experiment folder: %s' % (opts['exp_dir']))
    shutil.copy(args.exp, opts['exp_dir'])

    train_text = read_text(os.path.join(opts['training_path'], 'Train_text.xlsx'))
    train_tf = RandomGenerator(output_size=[opts['image_size'], opts['image_size']])
    train_dt = Mixdataset(dataset_path=opts['training_path'], row_text=train_text, joint_transform=train_tf,)
    val_text = read_text(os.path.join(opts['validation_path'], 'Val_text.xlsx'))
    val_tf = ValGenerator(output_size=[opts['image_size'], opts['image_size']])
    val_dt = Mixdataset(dataset_path=opts['validation_path'], row_text=val_text, joint_transform=val_tf,)
    test_text = read_text(os.path.join(opts['testing_path'], 'Test_text.xlsx'))
    test_tf = ValGenerator(output_size=[opts['image_size'], opts['image_size']])
    test_dt = Mixdataset(dataset_path=opts['testing_path'], row_text=test_text, joint_transform=test_tf)


    logger = logger_config(os.path.join(opts['exp_dir'], 'train.log'))
    train_loader = DataLoader(dataset=train_dt, batch_size=opts['batch_size'], shuffle=True,  drop_last=True)
    val_loader = DataLoader(dataset=val_dt, batch_size=opts['batch_size'], shuffle=False,  drop_last=True)
    
    # Use CustomDataset for test data loading
    # Define paths
    #img_folder = os.path.join(opts['testing_path'], 'img')
    #label_folder = os.path.join(opts['testing_path'], 'labelcol')
    #text_file = os.path.join(opts['testing_path'], 'Test_text.xlsx')
    
    #img_folder = "datasets/monuseg_2/Test_Folder/img"
    #label_folder = "datasets/monuseg_2/Test_Folder/labelcol"
    #text_file = "datasets/monuseg_2/Test_Folder/Test_text.xlsx"

    #test_dataset = CustomDataset(text_file, img_folder, label_folder, transform=None)
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    fea_extractor = create_feature_extractor(**opts)

    #main(opts, extractor=fea_extractor, data_loader=train_loader)
    # Initialize classifier before loading checkpoint
    
    extract_dims = [v * len(opts['steps']) for v in opts['dim']]
    classifier = pixel_classifier(extract_dims=extract_dims).to(device)  # Initialize model
    #classifier = classifier.to(device)  # Move model to the correct device
    #classifier.to("cpu")
    
    # Load trained model

    checkpoint_path = "/workspace/FastTextDiff/saved_textdiff/monuseg_2/experiment-02/model_best.pth"  
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Checkpoint keys:", checkpoint.keys())  # Debugging: Check keys in checkpoint
    if "model_state_dict" in checkpoint:
        classifier.load_state_dict(checkpoint["model_state_dict"])
    else:
        classifier.load_state_dict(checkpoint)  # Use if direct loading is needed
        
    classifier.eval()  # Set to evaluation mode
    print("Loaded trained model from:", checkpoint_path)
    
    
    # Define paths for single image inference
    img_path = "/workspace/FastTextDiff/datasets/monuseg_2/Test_Folder/img/TCGA-AC-A2FO-01A-01-TS1.tif"
    label_path = "/workspace/FastTextDiff/datasets/monuseg_2/Test_Folder/labelcol/TCGA-AC-A2FO-01A-01-TS1.png"
    text_file = "/workspace/FastTextDiff/datasets/monuseg_2/Test_Folder/Test_text.xlsx"

    # Run testing
    #print("Skipping training. Directly evaluating on test set...")
    #test(opts, classifier, fea_extractor, test_loader)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
    
    # Run inference only on single image
    infer_single_image(classifier, fea_extractor, img_path, label_path, text_file, tokenizer, bert_embedding, device)

    print("Inference complete.")
    
    