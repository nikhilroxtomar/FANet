
import os, time
from operator import add
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, accuracy_score

from model import FANet
from utils import create_dir, seeding, init_mask, rle_encode, rle_decode, load_data

def precision_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2_score(y_true, y_pred, beta=2):
    p = precision_score(y_true,y_pred)
    r = recall_score(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def calculate_metrics(y_true, y_pred, img):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_fbeta = F2_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    confusion = confusion_matrix(y_true, y_pred)
    if float(confusion[0,0] + confusion[0,1]) != 0:
        score_specificity = float(confusion[0,0]) / float(confusion[0,0] + confusion[0,1])
    else:
        score_specificity = 0.0

    return [score_jaccard, score_f1, score_recall, score_precision, score_specificity, score_acc, score_fbeta]

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

class CustomDataParallel(torch.nn.DataParallel):
	""" A Custom Data Parallel class that properly gathers lists of dictionaries. """
	def gather(self, outputs, output_device):
		# Note that I don't actually want to convert everything to the output_device
		return sum(outputs, [])

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Load dataset """
    path = "/media/nikhil/Seagate Backup Plus Drive/ML_DATASET/Kvasir-SEG"
    (train_x, train_y), (test_x, test_y) = load_data(path)

    """ Hyperparameters """
    size = (256, 256)
    num_iter = 10
    checkpoint_path = "files/checkpoint.pth"

    """ Directories """
    create_dir("results")

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FANet()
    # model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = CustomDataParallel(model).to(device)
    model.eval()

    """ Testing """
    prev_masks = init_mask(test_x, size)
    save_data = []
    file = open("files/test_results.csv", "w")
    file.write("Iteration,Jaccard,F1,Recall,Precision,Specificity,Accuracy,F2,Mean Time,Mean FPS\n")

    for iter in range(num_iter):

        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tmp_masks = []
        time_taken = []

        for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
            ## Image
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            image = cv2.resize(image, size)
            img_x = image
            image = np.transpose(image, (2, 0, 1))
            image = image/255.0
            image = np.expand_dims(image, axis=0)
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
            image = image.to(device)

            ## Mask
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, size)
            mask = np.expand_dims(mask, axis=0)
            mask = mask/255.0
            mask = np.expand_dims(mask, axis=0)
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
            mask = mask.to(device)

            ## Prev mask
            pmask = prev_masks[i]
            pmask = " ".join(str(d) for d in pmask)
            pmask = str(pmask)
            pmask = rle_decode(pmask, size)
            pmask = np.expand_dims(pmask, axis=0)
            pmask = np.expand_dims(pmask, axis=0)
            pmask = pmask.astype(np.float32)
            if iter == 0:
                pmask = np.transpose(pmask, (0, 1, 3, 2))
            pmask = torch.from_numpy(pmask)
            pmask = pmask.to(device)

            with torch.no_grad():
                """ FPS Calculation """
                start_time = time.time()
                pred_y = torch.sigmoid(model([image, pmask]))
                end_time = time.time() - start_time
                time_taken.append(end_time)

                score = calculate_metrics(mask, pred_y, img_x)
                metrics_score = list(map(add, metrics_score, score))
                pred_y = pred_y[0][0].cpu().numpy()
                pred_y = pred_y > 0.5
                pred_y = np.transpose(pred_y, (1, 0))
                pred_y = np.array(pred_y, dtype=np.uint8)
                pred_y = rle_encode(pred_y)
                prev_masks[i] = pred_y
                tmp_masks.append(pred_y)

        """ Mean Metrics Score """
        jaccard = metrics_score[0]/len(test_x)
        f1 = metrics_score[1]/len(test_x)
        recall = metrics_score[2]/len(test_x)
        precision = metrics_score[3]/len(test_x)
        specificity = metrics_score[4]/len(test_x)
        acc = metrics_score[5]/len(test_x)
        f2 = metrics_score[6]/len(test_x)

        """ Mean Time Calculation """
        mean_time_taken = np.mean(time_taken)
        print("Mean Time Taken: ", mean_time_taken)
        mean_fps = 1/mean_time_taken

        print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Specificity: {specificity:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - Mean Time: {mean_time_taken:1.7f} - Mean FPS: {mean_fps:1.7f}")

        save_str = f"{iter+1},{jaccard:1.4f},{f1:1.4f},{recall:1.4f},{precision:1.4f},{specificity:1.4f},{acc:1.7f},{f2:1.7f},{mean_time_taken:1.7f},{mean_fps:1.7f}\n"
        file.write(save_str)

        save_data.append(tmp_masks)
    save_data = np.array(save_data)

    """ Saving the masks. """
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        # mask = mask / 255
        # mask = (mask > 0.5) * 255
        mask = mask_parse(mask)

        name = y.split("/")[-1].split(".")[0]
        sep_line = np.ones((size[0], 10, 3)) * 128
        tmp = [image, sep_line, mask]

        for data in save_data:
            tmp.append(sep_line)
            d = data[i]
            d = " ".join(str(z) for z in d)
            d = str(d)
            d = rle_decode(d, size)
            d = d * 255
            d = mask_parse(d)

            tmp.append(d)

        cat_images = np.concatenate(tmp, axis=1)
        cv2.imwrite(f"results/{name}.png", cat_images)
