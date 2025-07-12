import numpy as np
import torch
from sklearn import preprocessing

class DataLoader(object):
    def __init__(self, opt):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # read malicious traffic and benign traffic
        if opt.dataset == 'botiot':
            malicious_content = np.load(f"./data/botiot/{opt.split}.npz", allow_pickle=True)  # change to the root path
            benign_content = np.load("./data/botiot/benign.npz", allow_pickle=True)  # change to the root path
        elif opt.dataset == 'cicids':
            malicious_content = np.load(f"./data/cicids/{opt.split}.npz", allow_pickle=True)  # change to the root path
            benign_content = np.load("./data/cicids/benign.npz", allow_pickle=True)  # change to the root path
        else:
            print("You need to add new dataset in the dataloader.py")
            exit(0)
        # benign traffic feature and label
        bx_train = benign_content['bx_train']
        by_train = benign_content['by_train']
        bx_test = benign_content['bx_test']
        by_test = benign_content['by_test']

        # malicious traffic feature and label
        train_feature = malicious_content['train_feature']
        train_label = malicious_content['train_label'] + 1
        test_seen_feature = malicious_content['test_seen_feature']
        test_seen_label = malicious_content['test_seen_label'] + 1
        test_unseen_feature = malicious_content['test_unseen_feature']
        test_unseen_label = malicious_content['test_unseen_label'] + 1
        traffic_names = malicious_content['attacks']

        # preprocessing
        detector_scaler = preprocessing.MinMaxScaler()
        discriminator_scaler = preprocessing.MinMaxScaler()
        classifier_scaler = preprocessing.MinMaxScaler()

        combined_train_feature = np.concatenate((bx_train, train_feature), axis=0)
        combined_test_feature = np.concatenate((bx_test, test_seen_feature, test_unseen_feature), axis=0)

        combined_train_feature = detector_scaler.fit_transform(combined_train_feature)
        combined_test_feature = detector_scaler.transform(combined_test_feature)

        # all features for training detector and inference(including benign traffic features)
        self.train_feature = torch.tensor(combined_train_feature, dtype=torch.float32).to(self.device)
        self.test_feature = torch.tensor(combined_test_feature, dtype=torch.float32).to(self.device)

        # all labels for training detector(including benign traffic features)
        self.train_label = torch.tensor(np.concatenate((by_train, train_label), axis=0), dtype=torch.long).to(self.device)
        self.test_label = torch.tensor(np.concatenate((by_test, test_seen_label, test_unseen_label), axis=0),
                                       dtype=torch.long).to(self.device)

        # all binary labels for training detector(including benign traffic features, 1 for benign and 2 for malicious)
        binary_by_train = torch.zeros(by_train.shape[0], dtype=torch.long).to(self.device)
        binary_by_test = torch.zeros(by_test.shape[0], dtype=torch.long).to(self.device)
        binary_ay_train = torch.ones(train_label.shape[0], dtype=torch.long).to(self.device)
        binary_ay_test = torch.ones(test_seen_label.shape[0] + test_unseen_label.shape[0], dtype=torch.long).to(self.device)
        self.binary_train = torch.cat((binary_by_train, binary_ay_train), dim=0)
        self.binary_test = torch.cat((binary_by_test, binary_ay_test), dim=0)

        # all malicious traffic features for training discriminator and inference
        combined_all_malicious_feature = np.concatenate((train_feature, test_seen_feature, test_unseen_feature), axis=0)
        self.all_malicious_feature = torch.tensor(discriminator_scaler.fit_transform(combined_all_malicious_feature),
                                                  dtype=torch.float32).to(self.device)

        # all binary malicious traffic labels for training discriminator(0 for seen and 1 for unseen)
        binary_ky_test = torch.zeros(test_seen_label.shape[0], dtype=torch.long).to(self.device)
        binary_uy_test = torch.ones(test_unseen_label.shape[0], dtype=torch.long).to(self.device)
        self.test_seen_unseen_label = torch.cat((binary_ky_test, binary_uy_test), dim=0)

        # all malicious traffic features for training classifier
        self.train_seen_feature = torch.tensor(classifier_scaler.fit_transform(train_feature),
                                               dtype=torch.float32).to(self.device)
        self.test_seen_feature = torch.tensor(classifier_scaler.transform(test_seen_feature),
                                              dtype=torch.float32).to(self.device)
        self.test_unseen_feature = torch.tensor(classifier_scaler.transform(test_unseen_feature),
                                                dtype=torch.float32).to(self.device)

        # all malicious traffic labels for training detector
        self.train_seen_label = torch.tensor(train_label, dtype=torch.long).to(self.device)
        self.test_seen_label = torch.tensor(test_seen_label, dtype=torch.long).to(self.device)
        self.test_unseen_label = torch.tensor(test_unseen_label, dtype=torch.long).to(self.device)

        # seen, unseen, malicious, all traffic categories
        self.seen_classes = torch.unique(self.test_seen_label)
        self.unseen_classes = torch.unique(self.test_unseen_label)
        self.malicious_classes = torch.cat((self.seen_classes, self.unseen_classes), dim=0)
        self.all_classes = torch.unique(self.test_label)

        # number of samples in training set and test set
        self.num_train = self.train_feature.size()[0]
        self.num_test = self.test_feature.size()[0]
        # size of benign traffic
        self.benign_size_test = by_test.shape[0]
        # all categories name
        self.traffic_names = traffic_names
        # print details of the dataset
        all_labels = torch.cat((self.train_label, self.test_label), dim=0)
        unique_labels, counts = torch.unique(all_labels, return_counts=True)
        print("Dataset details")
        for i, count in zip(unique_labels, counts):
            print(f'{self.traffic_names[i]}: {count.item()} samples')
        print("Seen malicious traffic categories:", self.traffic_names[self.seen_classes.cpu()])
        print("Unseen malicious traffic categories:", self.traffic_names[self.unseen_classes.cpu()])
