import os
import cv2
import numpy as np
import pandas as pd
import torch
import json
from sklearn.metrics import jaccard_score
from src.feature_extractors import create_feature_extractor
from src.pixel_classifier import pixel_classifier
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1)  # Convert to CxHxW format
    return image

def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.0
    mask = torch.tensor(mask).unsqueeze(0)  # Add channel dimension
    return mask

def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = labs.astype(np.float32)
    tmp_3dunet = predict_save.astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save, (448, 448))
        predict_save = cv2.resize(predict_save, (2000, 2000))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path, predict_save * 255)
    else:
        cv2.imwrite(save_path, predict_save * 255)
    
    # Draw bounding box
    fig, ax = plt.subplots()
    plt.gca().add_patch(patches.Rectangle(xy=(4, 4), width=120, height=20, color="white", linewidth=1))
    plt.imshow(predict_save * 255, cmap='gray')
    plt.axis("off")
    height, width = predict_save.shape
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path, dpi=2000)
    plt.close()
    
    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, text, img_RGB, labs, vis_save_path):
    model.eval()
    output = model(input_img.cuda(), text.cuda())
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path + '_predict.jpg')
    return dice_pred_tmp, iou_tmp

def main():
    # Load configuration from JSON file
    with open('experiments/monuseg_2/ddpm.json', 'r') as f:
        config = json.load(f)

    img_folder = config['testing_path'] + '/img'
    mask_folder = config['testing_path'] + '/labelcol'
    text_file = config['testing_path'] + '/Test_text.xlsx'
    save_folder = 'results'
    os.makedirs(save_folder, exist_ok=True)

    # Load text features from the Excel file
    text_features_df = pd.read_excel(text_file, index_col=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config['model_type']
    steps = config['steps']
    blocks = config['blocks']
    batch_size = config.get('batch_size', 1)
    model_path = config['model_path']
    input_activations = config['input_activations']
    image_size = config['image_size']
    #dim = config.get('dim', [3072, 1536, 768, 384])
    dim = config.get('dim', [512, 512, 256, 256])
    class_cond = config.get('class_cond', False)  # Default to False if not specified
    learn_sigma = config.get('learn_sigma', True)
    num_channels = config.get('num_channels', 256)
    num_res_blocks = config.get('num_res_blocks', 2)
    channel_mult = config.get('channel_mult', (1,2,4,8))  
    channel_mult_str = ",".join(map(str, channel_mult))  # Convert tuple to comma-separated string
    num_heads = config.get('num_heads', 16)
    num_head_channels = num_channels // num_heads
    num_heads_upsample = config.get('num_heads_upsample', -1)
    attention_resolutions = config.get('attention_resolutions', (32, 16, 8))
    attention_resolutions_str = ",".join(map(str, attention_resolutions))  # Convert tuple to comma-separated string
    dropout = config.get('dropout', 0.1)
    diffusion_steps = config.get('diffusion_steps', 1000)
    noise_schedule = config.get('noise_schedule', 'linear')
    timestep_respacing = config.get('timestep_respacing', '')
    use_kl = config.get('use_kl', False)
    predict_xstart = config.get('predict_xstart', False)
    rescale_timesteps = config.get('rescale_timesteps', False)
    rescale_learned_sigmas = config.get('rescale_learned_sigmas', False)
    use_checkpoint = config.get('use_checkpoint', False)
    use_scale_shift_norm = config.get('use_scale_shift_norm', True)
    resblock_updown = config.get('resblock_updown', False)
    use_fp16 = config.get('use_fp16', True)
    use_new_attention_order = config.get('use_new_attention_order', False)
    extractor = create_feature_extractor(model_type, steps=steps, blocks=blocks, batch_size=batch_size,model_path=model_path, input_activations=input_activations, image_size=image_size, class_cond=class_cond, learn_sigma=learn_sigma, num_channels=num_channels, num_res_blocks=num_res_blocks, channel_mult=channel_mult_str, num_heads=num_heads, num_head_channels=num_head_channels, num_heads_upsample=num_heads_upsample, attention_resolutions=attention_resolutions_str, dropout=dropout, diffusion_steps=diffusion_steps, noise_schedule=noise_schedule, timestep_respacing=timestep_respacing, use_kl=use_kl, predict_xstart=predict_xstart, rescale_timesteps=rescale_timesteps, rescale_learned_sigmas=rescale_learned_sigmas, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, resblock_updown=resblock_updown, use_fp16=use_fp16, use_new_attention_order=use_new_attention_order,dim=dim).to(device)
    extract_dims = config['dim']  # Use dim from config
    classifier = pixel_classifier().to(device)
    classifier.load_state_dict(
    torch.load('/workspace/TextDiff/saved_textdiff/monuseg_2/experiment-04/model_best.pth'), strict=False
)


    for img_name in os.listdir(img_folder):
        if img_name.endswith('.tif'):
            img_path = os.path.join(img_folder, img_name)
            mask_path = os.path.join(mask_folder, img_name.replace('.tif', '.png'))
            img = load_image(img_path).to(device)
            mask = load_mask(mask_path).to(device)
            
            # Extract text features for the current image
            text_features = text_features_df.loc[img_name].values
            text = torch.tensor(text_features, dtype=torch.float32).to(device)
            
            dice_pred, iou_pred = vis_and_save_heatmap(classifier, img, text, img, mask, os.path.join(save_folder, img_name))
            print(f'Image: {img_name}, Dice: {dice_pred:.3f}, IoU: {iou_pred:.3f}')

if __name__ == '__main__':
    main()