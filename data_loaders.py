import random
import operator
import os
from collections import OrderedDict
from PIL import Image
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler, BatchSampler

class MLCBaseDataset(Dataset):
    def __init__(self, data_path, train=True, label_order='freq2rare', feature_mean=None, feature_variance=None, label_vocabs=None, rescale=True, l2norm=True, is_sparse_data=True):
        self.label_card = -1
        self.feature_dim = -1
        self.label_dim = -1
        self.start_label = 0
        self.stop_label = -1
        self.max_labelset_size = -1
        self.feature_mean = feature_mean
        self.feature_variance = feature_variance
        self.label_order = label_order
        self.rescale = rescale
        self.l2norm = l2norm
        self.is_sparse_data = is_sparse_data

        self._set_data_file_name(data_path)
        self.is_train = train
        self._load_data_split()

        if self.is_train and label_vocabs is None:
            label_vocabs = self._build_label_vocab()

        self.label_vocab, self.label_ivocab = label_vocabs

        assert self.label_dim <= len(self.label_vocab), '{}:{}'.format(self.label_dim, len(self.label_vocab))
        self.label_dim = len(self.label_vocab)
        if self.label_order == 'fixed-random':
            prefix = np.array([self.label_vocab['BOS'], self.label_vocab['STOP']])
            self.rand_perm = np.random.permutation(range(2, self.label_dim))
            self.rand_perm = np.concatenate([prefix, self.rand_perm])

    def _set_data_file_name(self, data_path):
        raise NotImplementedError()

    def _load_data_split(self):
        raise NotImplementedError()

    def _load(self, data_filepath):
        rows, cols, values = [], [], []

        all_labels = []
        with open(data_filepath) as fin:
            metadata = fin.readline()

            assert len(metadata.split()) == 3

            num_total_instances, feature_dim, label_dim = metadata.split()
            num_total_instances = int(num_total_instances)
            feature_dim = int(feature_dim)
            label_dim = int(label_dim)

            self.label_dim = label_dim         # Number of the original labels
            self.feature_dim = feature_dim

            for idx, line in enumerate(fin):
                separator_idx = line.find(' ')
                if separator_idx == 0:
                    labels = []  # no label
                elif separator_idx > 0:
                    labels = line[:separator_idx].split(',')
                    # labels = ['label_' + str(label) for label in labels]
                    labels = [int(label) for label in labels]

                all_labels.append(labels)

                features = line[separator_idx+1:].split()

                for feat in features:
                    col_idx, val = feat.split(':')

                    rows.append(idx)
                    cols.append(col_idx)
                    values.append(val)

            all_features = coo_matrix((values, (rows, cols)),
                                      shape=(num_total_instances, feature_dim),
                                      dtype=float).tocsr()

        if self.rescale and self.feature_variance is None:
            self.feature_scaling(all_features)

        return all_features, all_labels

    def __len__(self):
        if self.is_train:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]

    def __getitem__(self, index):
        if self.is_train:
            data, labels = self.train_data[index], self.train_labels[index]
        else:
            data, labels = self.test_data[index], self.test_labels[index]

        if self.is_sparse_data:
            if self.rescale:
                data = data * sparse.diags(1/np.sqrt(self.feature_variance[0]))

            if self.l2norm:
                data = data / np.sqrt(np.sum(data.power(2), 1))

            I, J, V = sparse.find(data)
            data = (I, J, V)

        else:
            # data = np.squeeze(data.toarray(), axis=0).astype(float)
            data = data.toarray().astype(float)
            data = (data - self.feature_mean) / np.sqrt(self.feature_variance)
            data = np.squeeze(data, axis=0)
            # data = torch.FloatTensor(data)

        # convert a label subset into indices!
        transformed_labels = [self.label_vocab['label_' + str(label)] for label in labels]

        if self.label_order == 'freq2rare':
            transformed_labels.sort()
        elif self.label_order == 'rare2freq':
            transformed_labels.sort(reverse=True)
        elif self.label_order == 'same':
            # do nothing
            pass
        elif self.label_order == 'mblp':
            # do nothing
            pass
        elif self.label_order == 'always-random':
            random.shuffle(transformed_labels)
        elif self.label_order == 'fixed-random':
            labels_ = [(label, self.rand_perm[label]) for label in transformed_labels]
            transformed_labels = [label for (label, index) in sorted(labels_, key=operator.itemgetter(1))]
        else:
            raise ValueError('{} is not supported!'.format(self.label_order))

        labels = transformed_labels + [self.label_vocab['STOP']]

        return data, labels

    def feature_scaling(self, features):
        copied_features = features.copy()
        copied_features.data **= 2      
        feature_second_moment = np.array(copied_features.mean(axis=0))

        feature_first_moment = np.array(features.mean(axis=0))

        self.feature_mean = feature_first_moment.copy()
        self.feature_variance = feature_second_moment - feature_first_moment * feature_first_moment
        self.feature_variance[self.feature_variance == 0] = 1

        del copied_features, feature_first_moment, feature_second_moment

    def _build_label_vocab(self):
        assert self.is_train

        vocab = dict()

        # build a label vocabulary
        for labels in self.train_labels:
            for label in labels:
                assert type(label) == int
                if not label in vocab:
                    vocab[label] = 0

                vocab[label] += 1

        for label in range(self.label_dim):
            if not label in vocab:
                vocab[label] = 0

        assert len(vocab) == self.label_dim

        # sort the vocabulary by frequence in a descending order
        import operator
        sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)

        label_vocab = OrderedDict()
        label_ivocab = OrderedDict()

        # add predefined labels such as stop label
        label_vocab['BOS'] = int(0)
        label_vocab['STOP'] = int(1)
        for k, idx in label_vocab.items():
            label_ivocab[idx] = k

        idx = len(label_vocab)

        for (label, freq) in sorted_vocab:
            assert label not in label_vocab
            assert idx not in label_ivocab

            label_vocab['label_' + str(label)] = idx
            label_ivocab[idx] = 'label_' + str(label)

            idx += 1

        del vocab, sorted_vocab

        return label_vocab, label_ivocab
        
    def get_num_labels(self):
        assert self.label_dim == len(self.label_vocab)

        return self.label_dim

    def get_feature_dim(self):
        return self.feature_dim

    def get_label_cardinality(self):
        return self.label_card

    def get_max_labelset_size(self):
        return self.max_labelset_size

    def get_stop_label_id(self):
        return self.label_vocab['STOP']

    def get_start_label_id(self):
        return self.label_vocab['BOS']

    def get_feature_mean(self):
        return self.feature_mean

    def get_feature_variance(self):
        return self.feature_variance

    def get_label_vocabs(self):
        return (self.label_vocab, self.label_ivocab)

    def is_sparse_dataset(self):
        return self.is_sparse_data


class CVBaseDataset(MLCBaseDataset):
    def __init__(self, data_path, train=True, label_order='freq2rare', fold=0, feature_mean=None, feature_variance=None, label_vocabs=None, rescale=True, l2norm=True, is_sparse_data=True):
        self.fold = fold
        super(CVBaseDataset, self).__init__(data_path, train, label_order, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs, rescale=rescale, l2norm=l2norm, is_sparse_data=is_sparse_data)

    def _load_data_split(self):
        all_features, all_labels = self._load(self.data_filepath)

        if self.is_train:
            self.train_data, self.train_labels = self._choose_data_fold(self.train_split, all_features, all_labels, self.fold)
            labelset_size = list(map(len, self.train_labels))
            self.label_card = np.array(labelset_size).mean()
            self.max_labelset_size = np.array(labelset_size).max()
        else:
            self.test_data, self.test_labels = self._choose_data_fold(self.test_split, all_features, all_labels, self.fold)

        del all_features, all_labels

    def _choose_data_fold(self, data_split_filepath, all_features, all_labels, fold):
        data_split = np.loadtxt(data_split_filepath, dtype=int) - 1     # for 0-base indexing
        selected_fold = data_split[:, fold]

        selected_features_ = all_features[selected_fold, :]
        selected_labels_ = np.array(all_labels)[selected_fold]

        non_empty_indices = np.array([len(t)>0 for t in selected_labels_])
        selected_features = selected_features_[non_empty_indices, :]
        selected_labels = selected_labels_[non_empty_indices]

        return selected_features, selected_labels


class DeliciousDataset(CVBaseDataset):
    def __init__(self, data_path, train=True, label_order='freq2rare', fold=0, feature_mean=None, feature_variance=None, label_vocabs=None):
        super(DeliciousDataset, self).__init__(data_path, train, label_order, fold, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs, rescale=False, l2norm=True)

    def _set_data_file_name(self, data_path):
        self.data_filepath = os.path.join(data_path, 'Delicious_data.txt')
        self.train_split = os.path.join(data_path, 'delicious_trSplit.txt')
        self.test_split = os.path.join(data_path, 'delicious_tstSplit.txt')


class MediamillDataset(CVBaseDataset):
    def __init__(self, data_path, train=True, label_order='freq2rare', fold=0, feature_mean=None, feature_variance=None, label_vocabs=None):
        super(MediamillDataset, self).__init__(data_path, train, label_order, fold, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs, rescale=True, l2norm=False, is_sparse_data=False)

    def _set_data_file_name(self, data_path):
        self.data_filepath = os.path.join(data_path, 'Mediamill_data.txt')
        self.train_split = os.path.join(data_path, 'mediamill_trSplit.txt')
        self.test_split = os.path.join(data_path, 'mediamill_tstSplit.txt')


class BibtexDataset(CVBaseDataset):
    def __init__(self, data_path, train=True, label_order='freq2rare', fold=0, feature_mean=None, feature_variance=None, label_vocabs=None):
        super(BibtexDataset, self).__init__(data_path, train, label_order, fold, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs, rescale=False, l2norm=True)

    def _set_data_file_name(self, data_path):
        self.data_filepath = os.path.join(data_path, 'Bibtex_data.txt')
        self.train_split = os.path.join(data_path, 'bibtex_trSplit.txt')
        self.test_split = os.path.join(data_path, 'bibtex_tstSplit.txt')


class EurlexDataset(MLCBaseDataset):
    def __init__(self, data_path, train=True, label_order='freq2rare', feature_mean=None, feature_variance=None, label_vocabs=None):
        super(EurlexDataset, self).__init__(data_path, train, label_order, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs, rescale=False, l2norm=True)

    def _set_data_file_name(self, data_path):
        self.train_split = os.path.join(data_path, 'eurlex_train.txt')
        self.test_split = os.path.join(data_path, 'eurlex_test.txt')

    def _load_data_split(self):
        if self.is_train:
            self.train_data, self.train_labels = self._load(self.train_split)
            labelset_size = list(map(len, self.train_labels))
            self.label_card = np.array(labelset_size).mean()
            self.max_labelset_size = np.array(labelset_size).max()
        else:
            self.test_data, self.test_labels = self._load(self.test_split)


class RCVXDataset(MLCBaseDataset):
    def __init__(self, data_path, train=True, label_order='freq2rare', feature_mean=None, feature_variance=None, label_vocabs=None):
        super(RCVXDataset, self).__init__(data_path, train, label_order, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs)

    def _set_data_file_name(self, data_path):
        self.train_split = os.path.join(data_path, 'rcv1x_train.txt')
        self.test_split = os.path.join(data_path, 'rcv1x_test.txt')

    def _load_data_split(self):
        if self.is_train:
            self.train_data, self.train_labels = self._load(self.train_split)
            labelset_size = list(map(len, self.train_labels))
            self.label_card = np.array(labelset_size).mean()
            self.max_labelset_size = np.array(labelset_size).max()
        else:
            self.test_data, self.test_labels = self._load(self.test_split)


class AmazonCatDataset(MLCBaseDataset):
    def __init__(self, data_path, train=True, label_order='freq2rare', feature_mean=None, feature_variance=None, label_vocabs=None):
        super(AmazonCatDataset, self).__init__(data_path, train, label_order, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs)

    def _set_data_file_name(self, data_path):
        self.train_split = os.path.join(data_path, 'amazonCat_train.txt')
        self.test_split = os.path.join(data_path, 'amazonCat_test.txt')

    def _load_data_split(self):
        if self.is_train:
            self.train_data, self.train_labels = self._load(self.train_split)
            labelset_size = list(map(len, self.train_labels))
            self.label_card = np.array(labelset_size).mean()
            self.max_labelset_size = np.array(labelset_size).max()
        else:
            self.test_data, self.test_labels = self._load(self.test_split)


class Wiki10Dataset(MLCBaseDataset):
    def __init__(self, data_path, train=True, label_order='freq2rare', feature_mean=None, feature_variance=None, label_vocabs=None):
        super(Wiki10Dataset, self).__init__(data_path, train, label_order, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs, rescale=False)

    def _set_data_file_name(self, data_path):
        self.train_split = os.path.join(data_path, 'wiki10_train.txt')
        self.test_split = os.path.join(data_path, 'wiki10_test.txt')

    def _load_data_split(self):
        if self.is_train:
            self.train_data, self.train_labels = self._load(self.train_split)
            labelset_size = list(map(len, self.train_labels))
            self.label_card = np.array(labelset_size).mean()
            self.max_labelset_size = np.array(labelset_size).max()
        else:
            self.test_data, self.test_labels = self._load(self.test_split)


class NUSWIDEDataset(MLCBaseDataset):
    def __init__(self, data_path, train=True, label_order='freq2rare', transforms=None, label_vocabs=None):
        super(NUSWIDEDataset, self).__init__(data_path, train, label_order, label_vocabs=label_vocabs)

        self.data_path = data_path
        self.transform = transforms

    def _set_data_file_name(self, data_path):
        self.train_img_list = os.path.join(data_path, 'ImageList/TrainImagelist.txt')
        self.test_img_list = os.path.join(data_path, 'ImageList/TestImagelist.txt')

        self.train_tag_list = os.path.join(data_path, 'NUS_WID_Tags/Train_Tags81.txt')
        self.test_tag_list = os.path.join(data_path, 'NUS_WID_Tags/Test_Tags81.txt')

    def _load_data_split(self):
        if self.is_train:
            self.data, self.labels = self._load([self.train_img_list, self.train_tag_list])
            self.label_vocab = dict()
            for labelset in self.labels:
                for label in labelset:
                    if label not in self.label_vocab[label]:
                        self.label_vocab[label] = 0
                    self.label_vocab[label] += 1

            self.label_dim = len(self.label_vocab) + 2
            labelset_size = list(map(len, self.labels))
            self.label_card = np.array(labelset_size).mean()
            self.max_labelset_size = np.array(labelset_size).max()
        else:
            self.data, self.labels = self._load(self.test_split)

    def _load(self, filepaths):
        # data contains only filenames
        img_filepath, label_filepath = filepaths

        label_matrix = np.loadtxt(label_filepath, dtype=np.int64)

        num_instances = label_matrix.shape[0]
        all_labels = [[]] * num_instances

        for row, col in zip(*label_matrix.nonzero()):
            all_labels[row].append(col + 1)

        with open(img_filepath) as f:
            all_images = [os.path.join(*f_path.split('\\')).strip() for f_path in f]

        assert len(all_images) == num_instances

        return all_images, all_labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, 'Flickr', self.data[index]))
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)

        labelset = self.labels[index]

        return img, labelset

    def __len__(self):
        return len(self.data)


# def collate_fn(data):
#     """Creates mini-batch tensors from the list of tuples (image, caption).
# 
#     We should build custom collate_fn rather than using default collate_fn,
#     because merging caption (including padding) is not supported in default.
#     Args:
#         data: list of tuple (image, caption).
#             - image: torch tensor of shape (3, 256, 256).
#             - caption: torch tensor of shape (?); variable length.
#     Returns:
#         images: torch tensor of shape (batch_size, 3, 256, 256).
#         targets: torch tensor of shape (batch_size, padded_length).
#         lengths: list; valid length for each padded caption.
#     """
#     # Sort a data list by caption length (descending order).
#     # print(data)
#     # data.sort(key=lambda x: len(x[1]), reverse=True)
#     instances, labels = zip(*data)
# 
#     # Merge instances (from tuple of 1D tensor to 2D tensor).
#     instances = torch.stack(instances, 0)
# 
#     # Merge labels (from tuple of 1D tensor to 2D tensor).
#     # lengths = [len(label_set) for label_set in labels]
#     targets = []
#     for label_set in labels:
#         targets.append(label_set)
# 
#     return instances, targets
def collate_fn(data):
    instances, labels = zip(*data)
    print(len(instances))
    sys.exit(1)
    batch_size = len(instances)
    I = []
    J = []
    V = []
    for index in range(len(instances)):
        I.append(instances[index][0] + index)
        J.append(instances[index][1])
        V.append(instances[index][2])

    I = np.concatenate(I)
    J = np.concatenate(J)
    V = np.concatenate(V)

    I = np.vstack([I, J])

    instances = (I, V)

    # Merge labels (from tuple of 1D tensor to 2D tensor).
    labels = list(labels)

    return instances, labels

def non_sp_collate_fn(data):
    instances, labels = zip(*data)
    batch_size = len(instances)
    instances = np.stack(instances, 0)

    # Merge labels (from tuple of 1D tensor to 2D tensor).
    labels = list(labels)

    return instances, labels


class DatasetFactory(object):
    @staticmethod
    def get_dataset(name, train=True, label_order='freq2rare', fold=0, transforms=None, feature_mean=None, feature_variance=None, label_vocabs=None):
        dataset = None
        if name == 'eurlex':
            dataset = EurlexDataset('data/Eurlex', train=train, label_order=label_order, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs)
        elif name == 'rcv1':
            dataset = RCVXDataset('data/RCV1-x', train=train, label_order=label_order, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs)
        elif name == 'wiki10':
            dataset = Wiki10Dataset('data/Wiki10', train=train, label_order=label_order, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs)
        elif name == 'delicious':
            dataset = DeliciousDataset('data/Delicious', train=train, label_order=label_order, fold=fold, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs)
        elif name == 'mediamill':
            dataset = MediamillDataset('data/Mediamill', train=train, label_order=label_order, fold=fold, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs)
        elif name == 'bibtex':
            dataset = BibtexDataset('data/Bibtex', train=train, label_order=label_order, fold=fold, feature_mean=feature_meam, feature_variance=feature_variance, label_vocabs=label_vocabs)
        elif name == 'amazoncat':
            dataset = AmazonCatDataset('data/AmazonCat', train=train, label_order=label_order, feature_mean=feature_mean, feature_variance=feature_variance, label_vocabs=label_vocabs)
        elif name == 'nuswide':
            dataset = NUSWIDEDataset('data/NUS-WIDE', train=train, label_order=label_order, transforms=transforms, label_vocabs=label_vocabs)
        else:
            raise ValueError('No dataset available: %s'.format(name))

        return dataset


if __name__ == '__main__':
    shuffle = False
    random_seed = 1234

    batch_size = 128
    num_workers = 1
    pin_memory = False

    # transformations = transforms.Compose([
    #                                 transforms.Resize(256),
    #                                 transforms.CenterCrop(224),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                      std=[0.229, 0.224, 0.225])
    #                                 ])
    # train_dataset = DatasetFactory.get_dataset('nuswide', transforms=transformations)
    # valid_dataset = DatasetFactory.get_dataset('nuswide', transforms=transformations)
    is_sparse_data = False
    train_dataset = DatasetFactory.get_dataset('mediamill')
    valid_dataset = DatasetFactory.get_dataset('mediamill', label_order='freq2rare', feature_variance=train_dataset.get_feature_variance(),
                                               label_vocabs=train_dataset.get_label_vocabs())

    valid_size = 0.1

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if is_sparse_data:
        collate_fn = collate_fn
    else:
        collate_fn = non_sp_collate_fn

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SequentialSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, collate_fn=collate_fn
    )

    print('Number of labels: {}'.format(train_dataset.get_num_labels()))
    print('Feature dimensionality: {}'.format(train_dataset.get_feature_dim()))

    for epoch in range(10):
        for samples in train_loader:
            instances, labels = samples
            if is_sparse_data:
                batch_size = len(labels)
                I, V = instances
                instances = torch.sparse.FloatTensor(torch.LongTensor(I),
                        torch.FloatTensor(V), torch.Size([batch_size, train_dataset.get_feature_dim()]))
            else:
                instances = torch.FloatTensor(instances)

            print('Instance shape: {}'.format(instances.shape))
            print('done')
            sys.exit(1)

            if batch_idx % 10 == 0:
                print('At epoch {}, {}: {}'.format(epoch, batch_idx, len(data)))
