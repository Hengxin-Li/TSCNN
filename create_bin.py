import os
import glob
import random
import numpy as np
import cv2
import pickle
import torch
import torch.nn as nn
import torch.nn.parallel.data_parallel
import torch.backends.cudnn
import torch.optim.lr_scheduler
import torch.distributed
import torch.multiprocessing


def main(index_gpu):
    torch.cuda.empty_cache()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print('===> Creating dataloader...')
    DIV2K()
    print('===> Finish!')


class DIV2K(nn.Module):
    def __init__(self, name='div2k'):
        self.name = name
        self.data_pack = 'packet/packet'
        data_pack = self.data_pack.split('/')
        self.data_pack = data_pack[0]

        self.file_sort = FileNameSort()

        self.data_range = '1-900/1-100'  # train/test data range
        data_range = [r.split('-') for r in self.data_range.split('/')]

        self.train_range = data_range[0]
        self.n_train_begin = self.train_range[0]
        self.n_train_end = self.train_range[1]
        self.n_train_paris = int(self.n_train_end) - int(self.n_train_begin) + 1

        self.ext = ('png', 'bmp')
        self.flip = (1, 0, 0, 0)  # none, h, v, all
        self.rot = (1, 0, 0, 0)  # 0, 90, 180, 270

        self.dir_root = os.path.join('DATA', self.name)
         # '../DATA\\div2k'
        self.n_patch_size = 96  # x2:96 x3:144 x4:196

        self.n_step_size = self.n_patch_size
        self.n_data_paris = 0
        self.bin_label = []
        self.bin_data = []
        self.bin_x_dem = []

        self.scale = 2

        self.l_label = []
        self.l_data = []
        self.l_train_label = []
        self.l_train_data = []
        self.l_train_dem = []

        self.dir_train_label = os.path.join(self.dir_root, 'ori', 'train', 'label')
        self.dir_train_data = os.path.join(self.dir_root, 'ori', 'train', 'data')
        self.dir_train_dem = os.path.join(self.dir_root, 'ori', 'train', 'dem')

        for e in self.ext:
            self.l_train_label += glob.glob(os.path.join(self.dir_train_label, '*.' + e))
            self.l_train_data += glob.glob(os.path.join(self.dir_train_data, '*.' + e))
            self.l_train_dem += glob.glob(os.path.join(self.dir_train_dem, '*.' + e))
        self.l_train_label = self.file_sort.sort(self.l_train_label)
        self.l_train_data = self.file_sort.sort(self.l_train_data)
        self.l_train_dem = self.file_sort.sort(self.l_train_dem)
        self.l_train_label = self.l_train_label[int(self.n_train_begin) - 1:int(self.n_train_end)]
        self.l_train_data = self.l_train_data[int(self.n_train_begin) - 1:int(self.n_train_end)]
        self.l_train_dem = self.l_train_dem[int(self.n_train_begin) - 1:int(self.n_train_end)]

        self.make_pair()

        super(DIV2K, self).__init__()

    def __getitem__(self, index):
        label = self.bin_label[index]
        data = self.bin_data[index]

        return torch.from_numpy(data / 1.0).float(), torch.from_numpy(label / 1.0).float()

    def __len__(self):
        return self.n_data_paris

    def make_pair(self):
        n_dim = 0
        bin_label = []
        bin_dem = []
        bin_data = []
        for i in range(self.n_train_paris):
            filename, _ = os.path.splitext(os.path.basename(self.l_train_label[i]))
            # noinspection PyBroadException
            try:
                label = cv2.imread(self.l_train_label[i], cv2.IMREAD_UNCHANGED)
                label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
                data = cv2.imread(os.path.join(self.dir_train_data, filename + _), cv2.IMREAD_UNCHANGED)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                x_dem = cv2.imread(self.l_train_dem[i], cv2.IMREAD_UNCHANGED)
                x_dem = cv2.cvtColor(x_dem, cv2.COLOR_BGR2RGB)
                if label is None or data is None or x_dem is None:
                    print("label or data or  x_dem is None.")
                    return
            except Exception as e:
                print(e)
                continue
            ih, iw = label.shape[:2]
            iw = iw // 4 * 4
            ih = ih // 4 * 4
            label = label[:ih, :iw, :]

            if len(label.shape[2:3]) != 0 and label.shape[2:3][0] >= 3:
                label = label[:, :, 0:3]
            else:
                pass
            if len(data.shape[2:3]) != 0 and data.shape[2:3][0] >= 3:
                data = data[:, :, 0:3]
            else:
                pass

            index = 1
            for j in range(0, ih, self.n_step_size):
                for k in range(0, iw, self.n_step_size):
                    try:
                        oir_pair_label = label[j:j + self.n_patch_size, k:k + self.n_patch_size, :]
                        oir_pair_dem = x_dem[j//2:(j + self.n_patch_size)//2, k//2:(k + self.n_patch_size)//2, :]
                        oir_pair_data = data[j//2:(j + self.n_patch_size)//2, k//2:(k + self.n_patch_size)//2, :]

                        if oir_pair_label.shape[:2] != (self.n_patch_size, self.n_patch_size):
                            continue

                        pair_label = oir_pair_label
                        pair_dem = oir_pair_dem
                        pair_data = oir_pair_data

                        for i_f, f in enumerate(self.flip):
                            if f == 1:
                                if i_f == 1:  # hflip
                                    pair_label = cv2.flip(oir_pair_label, 1)
                                    pair_dem = cv2.flip(pair_dem, 1)
                                    pair_data = cv2.flip(oir_pair_data, 1)
                                elif i_f == 2:  # vflip
                                    pair_label = cv2.flip(oir_pair_label, 0)
                                    pair_dem = cv2.flip(pair_dem, 0)
                                    pair_data = cv2.flip(oir_pair_data, 0)
                                elif i_f == 3:  # all
                                    pair_label = cv2.flip(oir_pair_label, -1)
                                    pair_dem = cv2.flip(pair_dem, -1)
                                    pair_data = cv2.flip(oir_pair_data, -1)

                                for i_r, r in enumerate(self.rot):
                                    if r == 1:
                                        temp_label = pair_label
                                        temp_dem = pair_dem
                                        temp_data = pair_data

                                        m = cv2.getRotationMatrix2D(((self.n_patch_size - 1) / 2.0,
                                                                     (self.n_patch_size - 1) / 2.0), 90 * i_r, 1)
                                        m2 = cv2.getRotationMatrix2D(((self.n_patch_size//2 - 1) / 2.0,
                                                                     (self.n_patch_size//2 - 1) / 2.0), 90 * i_r, 1)
                                        temp_label = cv2.warpAffine(temp_label, m, (self.n_patch_size, self.n_patch_size))
                                        temp_dem = cv2.warpAffine(temp_dem, m2, (self.n_patch_size // 2, self.n_patch_size // 2))
                                        temp_data = cv2.warpAffine(temp_data, m2, (self.n_patch_size//2, self.n_patch_size//2))

                                        n_dim += 1

                                        bin_label.append(np.ascontiguousarray(temp_label.transpose((2, 0, 1))))
                                        bin_dem.append(np.ascontiguousarray(temp_dem.transpose((2, 0, 1))))
                                        bin_data.append(np.ascontiguousarray(temp_data.transpose((2, 0, 1))))

                                        index += 1
                    except Exception as e:
                        print(e)
                        continue
        if self.data_pack == 'packet':
            self.bin_label = np.array(bin_label)
            self.bin_x_dem = np.array(bin_dem)
            self.bin_data = np.array(bin_data)
            os.makedirs(os.path.join(self.dir_root, 'bin', 'train'), exist_ok=True)
            np.savez(os.path.join(self.dir_root, 'bin', 'train', self.name + '_train'),
                     label=self.bin_label, x_dem=self.bin_x_dem, data=self.bin_data)  # savez_compressed


class FileNameSort(object):

    def _is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def _find_continuous_num(self, astr, c):
        num = ''
        # noinspection PyBroadException
        try:
            while not self._is_number(astr[c]) and c < len(astr):
                c += 1
            while self._is_number(astr[c]) and c < len(astr):
                num += astr[c]
                c += 1
        except:
            pass
        if num != '':
            return int(num)

    def _comp2filename(self, file1, file2):
        smaller_length = min(len(file1), len(file2))
        continuous_num = ''
        for c in range(0, smaller_length):
            if not self._is_number(file1[c]) and not self._is_number(file2[c]):
                # print('both not number')
                if file1[c] < file2[c]:
                    return True
                if file1[c] > file2[c]:
                    return False
                if file1[c] == file2[c]:
                    if c == smaller_length - 1:
                        # print('the last bit')
                        if len(file1) < len(file2):
                            return True
                        else:
                            return False
                    else:
                        continue
            if self._is_number(file1[c]) and not self._is_number(file2[c]):
                return True
            if not self._is_number(file1[c]) and self._is_number(file2[c]):
                return False
            if self._is_number(file1[c]) and self._is_number(file2[c]):
                if self._find_continuous_num(file1, c) < self._find_continuous_num(file2, c):
                    return True
                else:
                    return False

    def sort_insert(self, lst):
        for i in range(1, len(lst)):
            x = lst[i]
            j = i
            while j > 0 and lst[j - 1] > x:
                lst[j] = lst[j - 1]
                j -= 1
            lst[j] = x
        return lst

    def sort(self, lst):
        for i in range(1, len(lst)):
            x = lst[i]
            j = i
            while j > 0 and self._comp2filename(x, lst[j - 1]):
                lst[j] = lst[j - 1]
                j -= 1
            lst[j] = x
        return lst


if __name__ == '__main__':
    device = torch.device('cuda')
    main(1)
