from copy import deepcopy
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import argparse
import xgboost as xgb
from dataloader import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, confusion_matrix, \
    classification_report, roc_curve
import json
import umap
import matplotlib.colors as mcolors

# set cuda environment variable
# note: Please comment out this line of code if GPU is unavailable,
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


# set the fontsize
def get_fontsize(label):
    # return the fontsize according to the dataset type, default is 10
    if opt.dataset == 'cicids':
        if label == 'confusion':
            return 12
        elif label == 'manifold':
            return 15
        elif label == 'histogram':
            return 20
        else:
            return 10

    elif opt.dataset == 'botiot':
        if label == 'confusion':
            return 20
        elif label == 'manifold':
            return 15
        elif label == 'histogram':
            return 20
        else:
            return 10
    else:
        return 10

    # required functions


class ConfusionMatrix(object):
    # plot confusion matrix
    def __init__(self, num_classes: int, labels: list, highlight_indices: np.ndarray = None):
        self.num_classes = num_classes
        self.labels = labels
        self.highlight_indices = highlight_indices
        self.matrix = np.zeros((num_classes, num_classes))
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, preds, real):
        for p, t in zip(preds, real):
            self.matrix[p, t] += 1

    def plot(self):
        self.confusion_matrix = deepcopy(self.matrix)
        for i in range(len(self.matrix[0])):
            total_num = np.sum(self.matrix[:, i])
            for j in range(len(self.matrix[0])):
                # self.confusion_matrix[j][i] = round(self.confusion_matrix[j][i] / total_num, 3)
                self.confusion_matrix[j][i] = round(float(self.confusion_matrix[j][i]) / total_num,
                                                    2) if total_num > 0 else 0.0

        matrix = np.array(self.confusion_matrix)

        plt.figure(figsize=(10, 8))

        plt.imshow(matrix, cmap='Blues', aspect='auto')

        # set the fontsize
        fontsize = get_fontsize('confusion')

        plt.xticks(range(self.num_classes), self.labels, rotation=45, fontsize=fontsize)
        plt.yticks(range(self.num_classes), self.labels, fontsize=fontsize)

        # set the size of colorbar
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=fontsize)
        plt.xlabel('True Labels', fontsize=fontsize)
        plt.ylabel('Predicted Labels', fontsize=fontsize)

        # remove the ticks of the axis
        plt.gca().tick_params(axis='both', length=0)

        # mark probability
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = matrix[y, x]
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         fontsize=fontsize,
                         color="white" if info > thresh else "black")

        # highlight specified indices
        if self.highlight_indices is not None:
            for index in self.highlight_indices:
                plt.gca().get_xticklabels()[index].set_color('red')
                plt.gca().get_yticklabels()[index].set_color('red')

        result_dir = f'./result/{opt.dataset}/HIoT/hybrid/{opt.split}/'
        os.makedirs(result_dir, exist_ok=True)
        plt.savefig(result_dir + 'confusion_matrix.svg', bbox_inches='tight')
        plt.show()


def load_args(path="./args/args.json"):
    # load argparse parameters from json
    with open(path, "r") as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)


def map_label(label, classes):
    # transform label and make them continuous (like label 2, 5 ,7 transform to 2, 3, 4)
    mapped_label = torch.zeros_like(label, dtype=torch.long)
    for i, class_label in enumerate(classes):
        mapped_label[label == class_label] = i
    return mapped_label.to(device)


def inverse_map(label, classes):
    # inverse label transformation
    label = label.cpu().numpy() if hasattr(label, "cpu") else label
    mapped_label = np.zeros_like(label)
    classes = classes.cpu().numpy() if hasattr(classes, "cpu") else classes
    for i, class_label in enumerate(classes):
        mapped_label[label == i] = class_label
    return mapped_label


def random_sampling(feature, label, sample_ratio):
    num_samples = max(1, int(len(label) * sample_ratio))
    sampled_indices = torch.randperm(len(label))[:num_samples]
    return feature[sampled_indices], label[sampled_indices], sampled_indices


def manifold_visualization(score):
    # random sampling and concatenate
    sampled_seen_feature, sampled_seen_label, sampled_seen_indices = random_sampling(dataset.test_seen_feature,
                                                                                     dataset.test_seen_label,
                                                                                     5 * opt.factor)
    sampled_unseen_feature, sampled_unseen_label, sampled_unseen_indices = random_sampling(dataset.test_unseen_feature,
                                                                                           dataset.test_unseen_label,
                                                                                           opt.factor)
    features = torch.cat((sampled_seen_feature, sampled_unseen_feature), dim=0)

    # corresponding OOD score
    score_seen, score_unseen = torch.split(score, [len(dataset.test_seen_label), len(dataset.test_unseen_label)])
    sampled_score_seen = score_seen[sampled_seen_indices]
    sampled_score_unseen = score_unseen[sampled_unseen_indices]
    ood_scores = torch.cat((sampled_score_seen, sampled_score_unseen), dim=0)

    features = features.cpu().numpy()
    ood_scores = ood_scores.cpu().numpy()

    # feature dimensionality reduction with umap
    reducer = umap.UMAP(random_state=opt.manualSeed)
    embedding = reducer.fit_transform(features)

    # create and set colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ['#1F77B4', '#ED7D31'])
    num_seen = len(sampled_seen_label)
    num_unseen = len(embedding) - num_seen
    colors = np.concatenate([
        np.full(num_seen, 0),
        np.full(num_unseen, 1)
    ])

    # manifold of malicious traffic
    result_dir = f'./result/{opt.dataset}/HIoT/discrimination/{opt.split}/'
    os.makedirs(result_dir, exist_ok=True)
    fontsize = get_fontsize('manifold')
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap, alpha=0.6)
    cbar = plt.colorbar(scatter)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label('Class Label', fontsize=fontsize)
    plt.title('Visualization of Malicious Traffic', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(result_dir + 'manifold.svg', bbox_inches='tight')
    plt.show()

    # corresponding OOD score
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=ood_scores, cmap='Spectral_r', alpha=0.6)
    cbar = plt.colorbar(scatter, label='OOD Score')
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label('OOD Score', fontsize=fontsize)
    plt.title('Visualization of OOD Scores', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(result_dir + 'scores.svg', bbox_inches='tight')
    plt.show()


def histogram_plot(pred_score, stage):
    # set axis interval
    x_min = pred_score.min()
    x_max = pred_score.max()
    range_x = x_max - x_min
    x_interval = range_x / 5

    bins_hist = np.linspace(x_min, x_max, 26)
    bar_width = bins_hist[1] - bins_hist[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    fontsize = get_fontsize('histogram')

    # remove axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # set the main scale line color
    ax.tick_params(axis='both', length=0, labelcolor='#000000', labelsize=fontsize)

    # set the x-axis scale
    ax.set_xticks(np.arange(x_min, x_max + x_interval, x_interval))
    ax.set_xlim(x_min - 0.02 * range_x, x_max + 0.02 * range_x)

    # count the samples in each interval
    seen_unseen_label = dataset.test_seen_unseen_label.cpu().numpy()
    # detection
    if stage == 1:
        result_dir = f'./result/{opt.dataset}/HIoT/detection/{opt.split}/'
        benign_label = dataset.binary_test.cpu().numpy()
        benign_counts, _ = np.histogram(pred_score[benign_label == 0], bins=bins_hist)
        malicious = pred_score[benign_label == 1]
        seen_malicious_counts, _ = np.histogram(malicious[seen_unseen_label == 0], bins=bins_hist)
        unseen_malicious_counts, _ = np.histogram(malicious[seen_unseen_label == 1], bins=bins_hist)
        # normalization
        benign_counts = benign_counts / len(pred_score[benign_label == 0])
        seen_malicious_counts = seen_malicious_counts / len(malicious[seen_unseen_label == 0])
        unseen_malicious_counts = unseen_malicious_counts / len(malicious[seen_unseen_label == 1])
        # plot histogram of benign traffic
        ax.bar(bins_hist[:-1], benign_counts, width=bar_width, align='edge',
               color="#5B9BD5", alpha=0.5, label="Benign")
        # set labels and legend
        ax.set_xlabel("Anomaly Score", fontsize=fontsize)
        ax.set_ylabel("Density", fontsize=fontsize)
        ax.set_title("Density Histogram of the Detector", fontsize=fontsize)
    # discrimination
    elif stage == 2:
        result_dir = f'./result/{opt.dataset}/HIoT/discrimination/{opt.split}/'
        seen_malicious_counts, _ = np.histogram(pred_score[seen_unseen_label == 0], bins=bins_hist)
        unseen_malicious_counts, _ = np.histogram(pred_score[seen_unseen_label == 1], bins=bins_hist)
        # normalization
        seen_malicious_counts = seen_malicious_counts / len(pred_score[seen_unseen_label == 0])
        unseen_malicious_counts = unseen_malicious_counts / len(pred_score[seen_unseen_label == 1])
        # set labels and legend
        ax.set_xlabel("OOD Score", fontsize=fontsize)
        ax.set_ylabel("Density", fontsize=fontsize)
        ax.set_title("Density Histogram of the Discriminator", fontsize=fontsize)
    else:
        print("Please input the right stage")
        exit(0)

    # plot histogram of seen malicious traffic
    ax.bar(bins_hist[:-1], seen_malicious_counts, width=bar_width, align='edge',
           color="#70AD47", alpha=0.5, label="Seen Malicious")

    # plot histogram of unseen malicious traffic
    ax.bar(bins_hist[:-1], unseen_malicious_counts, width=bar_width, align='edge',
           color="#C00000", alpha=0.5, label="Unseen Malicious")

    # add grid lines
    ax.yaxis.grid(True, linestyle='--', linewidth=1.75, color='#D9D9D9')

    ax.legend(loc='upper right', bbox_to_anchor=(0.9, 1), fontsize=fontsize - 5)

    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(result_dir + 'histogram.svg', bbox_inches='tight')
    plt.show()


def importance_plot():
    result_dir = f'./result/{opt.dataset}/HIoT/detection/{opt.split}/'
    xgb.plot_importance(model.model, max_num_features=10)
    plt.savefig(result_dir + 'feature_importance.svg', bbox_inches='tight')


def evaluation(y_true, score, option):
    # print classification report
    score = np.array(score)
    if option == 3:
        report = classification_report(y_true, score,
                                       target_names=dataset.traffic_names[dataset.seen_classes.cpu().numpy()], digits=4)
        print(report)
        return

    # ultimate hybrid report
    if option == 4:
        report = classification_report(y_true, score, target_names=dataset.traffic_names, digits=4)
        print(report)
        return

    # calculate AUC-ROC
    auc_roc = roc_auc_score(y_true, score)
    print(f"AUC-ROC: {auc_roc}")

    # calculate Precision-Recall and AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, score)
    auc_pr = auc(recall, precision)
    print(f"AUC-PR: {auc_pr}")

    # choose the threshold through TPR[threshold] (Model Performance when True Positive Rate achieves the threshold like 0.95 or 0.99)
    _, tpr, thresholds = roc_curve(y_true, score)
    indices = np.argwhere(tpr >= opt.threshold)
    if len(indices) > 0:
        best_index = indices[0]
    else:
        best_index = np.argmax(tpr)
    best_threshold = thresholds[best_index]
    print(best_threshold)
    y_pred = (score >= best_threshold).astype(int)

    # calculate F1-Score
    f1 = f1_score(y_true, y_pred)
    print(f"F1-Score: {f1}")

    # calculate Micro Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    # calculate Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")

    # calculate class-wise Recall
    recall_per_class = {}
    y_pred = torch.from_numpy(y_pred).to(device)
    ave_recall = 0

    if option == 1:  # detector
        for cls in dataset.all_classes:
            if cls.item() == 0:
                tp = torch.sum(torch.logical_and(y_pred.eq(0), dataset.test_label.eq(0))).item()
                fn = torch.sum(torch.logical_and(y_pred.eq(1), dataset.test_label.eq(0))).item()
            else:
                tp = torch.sum(torch.logical_and(y_pred.eq(1), dataset.test_label.eq(cls))).item()
                fn = torch.sum(torch.logical_and(y_pred.eq(0), dataset.test_label.eq(cls))).item()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_per_class[cls.item()] = recall
            ave_recall += recall
        ave_recall /= dataset.all_classes.shape[0]

    elif option == 2:  # discrimination
        for cls in dataset.malicious_classes:
            if cls in dataset.seen_classes:
                tp = torch.sum(
                    torch.logical_and(y_pred.eq(0), dataset.test_label[dataset.benign_size_test:].eq(cls))).item()
                fn = torch.sum(
                    torch.logical_and(y_pred.eq(1), dataset.test_label[dataset.benign_size_test:].eq(cls))).item()
            else:
                tp = torch.sum(
                    torch.logical_and(y_pred.eq(1), dataset.test_label[dataset.benign_size_test:].eq(cls))).item()
                fn = torch.sum(
                    torch.logical_and(y_pred.eq(0), dataset.test_label[dataset.benign_size_test:].eq(cls))).item()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_per_class[cls.item()] = recall
            ave_recall += recall
        ave_recall /= dataset.malicious_classes.shape[0]

    print("Recall per class:")
    for cls, recall in recall_per_class.items():
        print(f"Class {dataset.traffic_names[cls]}: Recall = {recall}")

    return y_pred, best_threshold


parser = argparse.ArgumentParser()

# set hyperparameters
# note: For all CIC-IDS2017 dataset splits, use cicids_args.json in the args folder to get the reported result.
# note: For all Bot-Iot dataset splits, use botiot_args.json in the args folder to get the reported result.
parser.add_argument('--dataset', default='cicids', help='Dataset')
parser.add_argument('--split', default='1', help='Dataset split for training and evaluation')
parser.add_argument('--manualSeed', type=int, default=42, help='Random seed')
parser.add_argument('--threshold', type=float, default=0.99, help='Evaluate model performance with TPR[threshold]')
parser.add_argument('--factor', type=float, default=0.02, help='Scaling factor for umap visualization')
parser.add_argument('--visualize', type=bool, default=True, help='Whether to visualize the data')
opt = parser.parse_args()

# load pre-defined hyperparameters
# note: If you want to customize the hyperparameters, please comment out this line of code.
opt = load_args("args/HIoT/cicids_args.json")

# set seed
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# loading data
dataset = DataLoader(opt)

# set the device
device = dataset.device

# 1st step: Detect malicious traffic
from adbench.baseline.Supervised import supervised

model = supervised(seed=42, model_name='XGB')  # initialization

# training procedure of detector
print("start fitting detector")
start_time = time.time()
model.fit(dataset.train_feature.cpu().numpy(), dataset.binary_train.cpu().numpy())  # fit
end_time = time.time()
print("end fitting detector")
print("Training time of the detector：%.4f seconds" % (end_time - start_time))

# inference procedure of detector
print("start evaluating detector")
start_time = time.time()
detector_score = model.predict_score(dataset.test_feature.cpu().numpy())  # predict
end_time = time.time()
# evaluation of detector
y_true = dataset.binary_test.cpu().numpy()
detector_prediction, _ = evaluation(y_true, detector_score, 1)

if opt.visualize:
    # plot histogram
    histogram_plot(detector_score, 1)
    # plot feature importance score
    importance_plot()

print("end evaluating detector")
print("Inference time of the detector：%.4f seconds" % (end_time - start_time))

# 2nd step: Discriminate unseen categories traffic
test_feature = dataset.all_malicious_feature[dataset.train_seen_feature.shape[0]:]
# training procedure of discriminator
print("start fitting and evaluating discriminator")
seen_class_classifier = xgb.XGBClassifier()
seen_class_classifier.fit(dataset.train_seen_feature.cpu().numpy(),
                          map_label(dataset.train_seen_label, dataset.seen_classes).cpu().numpy())
proba = seen_class_classifier.predict_proba(test_feature.cpu().numpy())
discriminator_score = 1 / np.max(proba, axis=1)

discriminator_prediction, threshold = evaluation(dataset.test_seen_unseen_label.cpu().numpy(), discriminator_score, 2)

# plot histogram and manifold
if opt.visualize:
    histogram_plot(discriminator_score, 2)
    manifold_visualization(torch.from_numpy(discriminator_score).to(device))
print("end fitting and evaluating discriminator")

# 3rd step: Classify seen categories traffic
print("start fitting and evaluating classifier")
# inference procedure of seen_class_classifier
seen_preds = seen_class_classifier.predict(dataset.test_seen_feature.cpu().numpy())
# evaluation of seen_class_classifier
evaluation(map_label(dataset.test_seen_label, dataset.seen_classes).cpu().numpy(), seen_preds, 3)

print("end fitting and evaluating classifier")

unseen_preds = map_label(dataset.test_unseen_label, dataset.unseen_classes).cpu().numpy()

# 4th step: hybrid ultimate output
print("start calculating hybrid performance")
# inverse predictions for malicious traffic
seen_preds_inverse = inverse_map(seen_preds, dataset.seen_classes)
unseen_preds_inverse = inverse_map(unseen_preds, dataset.unseen_classes)

# collect all predictions (benign and malicious)
preds_all = np.concatenate(
    (detector_prediction[:dataset.benign_size_test].cpu().numpy(), seen_preds_inverse, unseen_preds_inverse), axis=0)

# get score for benign and malicious traffic
score_for_benign = detector_prediction[:dataset.benign_size_test]
score_for_malicious = detector_prediction[dataset.benign_size_test:]

# get score for seen and unseen malicious traffic
score_for_seen = discriminator_prediction[:dataset.test_seen_feature.shape[0]]
score_for_unseen = discriminator_prediction[dataset.test_seen_feature.shape[0]:]

# get index for wrongly detected benign traffic
det_wrong_benign = torch.where(score_for_benign.eq(1))[0].to(device)
# get index for undetected malicious traffic
det_wrong_malicious = torch.where(score_for_malicious.eq(0))[0].to(device)

# get index for wrongly discriminated seen malicious traffic
dis_wrong_seen = torch.where(score_for_seen.eq(1))[0].to(device)
# get index for wrongly discriminated unseen malicious traffic
dis_wrong_unseen = torch.where(score_for_unseen.eq(0))[0].to(device)

with torch.no_grad():
    # 1.give wrongly discriminated seen malicious traffic new labels from unseen classes
    if len(dis_wrong_seen) > 0:
        # random assignment for unseen classes
        corrected_seen_preds = torch.randint(low=0, high=dataset.unseen_classes.shape[0],
                                             size=(dis_wrong_seen.shape[0],))
        corrected_seen_preds_inverse = inverse_map(corrected_seen_preds, dataset.unseen_classes)
        preds_all[dataset.benign_size_test + dis_wrong_seen.cpu().numpy()] = corrected_seen_preds_inverse

    # 2. give wrongly discriminated unseen malicious traffic new labels from seen classes
    if len(dis_wrong_unseen) > 0:
        unseen_features = dataset.test_unseen_feature[dis_wrong_unseen]
        corrected_unseen_preds = seen_class_classifier.predict(unseen_features.cpu().numpy())
        corrected_unseen_preds_inverse = inverse_map(corrected_unseen_preds, dataset.seen_classes)
        preds_all[dataset.benign_size_test + dataset.test_seen_feature.shape[
            0] + dis_wrong_unseen.cpu().numpy()] = corrected_unseen_preds_inverse

    # 3. give wrongly detected benign traffic new labels from malicious classes
    if len(det_wrong_benign) > 0:
        test_benign_feature = dataset.test_feature[:dataset.benign_size_test]
        benign_features = test_benign_feature[det_wrong_benign]

        proba = seen_class_classifier.predict_proba(benign_features.cpu().numpy())
        benign_discriminator_scores = np.min(proba, axis=1)

        # random assignment for unseen classes
        corrected_benign_unseen_preds = torch.randint(low=0, high=dataset.unseen_classes.shape[0],
                                                      size=(benign_features.shape[0],))
        seen_unseen_preds = np.where(benign_discriminator_scores < threshold,
                                     inverse_map(seen_class_classifier.predict(benign_features.cpu().numpy()),
                                                 dataset.seen_classes),
                                     inverse_map(corrected_benign_unseen_preds, dataset.unseen_classes))

        preds_all[det_wrong_benign.cpu().numpy()] = seen_unseen_preds

    # 4. give undetected malicious traffic benign labels
    if len(det_wrong_malicious) > 0:
        preds_all[dataset.benign_size_test + det_wrong_malicious.cpu().numpy()] = 0

# 5. final evaluation
evaluation(dataset.test_label.cpu().numpy(), preds_all, 4)
traffic_names = dataset.traffic_names
if opt.dataset == 'cicids':
    traffic_names[3] = "GoldenEye"
    traffic_names[4] = "Hulk"
    traffic_names[5] = "Slowhttptest"
    traffic_names[6] = "Slowloris"
    traffic_names[-1] = "XSS"
    traffic_names[-2] = "Sql Injection"
    traffic_names[-3] = "Brute Force"
elif opt.dataset == 'botiot':
    traffic_names[-2] = "Scan"
confusion = ConfusionMatrix(num_classes=len(dataset.all_classes), labels=traffic_names,
                            highlight_indices=dataset.unseen_classes.cpu().numpy())
confusion.update(preds_all, dataset.test_label.cpu().numpy())

# visualize the confusion matrix
if opt.visualize:
    confusion.plot()
