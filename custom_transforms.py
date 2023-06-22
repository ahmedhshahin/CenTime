import torch
from torch.nn.functional import pad as torch_pad
from typing import Any, List, Sequence, Tuple
from torch import Tensor
import numbers
from PIL import Image
import numpy as np
import cv2

def _is_pil_image(img: Any) -> bool:
    return isinstance(img, Image.Image)

def _get_image_size(img: Tensor) -> List[int]:
    _assert_image_tensor(img)
    return [img.shape[-3], img.shape[-1], img.shape[-2]]

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0], size[0]

    if len(size) != 3:
        raise ValueError(error_msg)

    return size

def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2

def _assert_image_tensor(img):
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")

def _pad_symmetric(img: Tensor, padding: List[int]) -> Tensor:
    # padding is front, back, left, right, top, bottom

    # crop if needed
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0 or padding[4] < 0 or padding[5] < 0:
        crop_front, crop_back, crop_left, crop_right, crop_top, crop_bottom = [-min(x, 0) for x in padding]
        img = img[..., crop_front:img.shape[-3] - crop_back, crop_top:img.shape[-2] - crop_bottom, crop_left:img.shape[-1] - crop_right]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.size()

    z_indices = [i for i in range(in_sizes[-3])]
    front_indices = [i for i in range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    back_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]
    z_indices = torch.tensor(front_indices + z_indices + back_indices)

    x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in range(padding[2] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    right_indices = [-(i + 1) for i in range(padding[3])]  # e.g. [-1, -2, -3]
    x_indices = torch.tensor(left_indices + x_indices + right_indices)

    y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[4] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[5])]
    y_indices = torch.tensor(top_indices + y_indices + bottom_indices)

    ndim = img.ndim
    if ndim == 4:
        return img[:, z_indices[:, None], y_indices[:, None], x_indices[None, :]]
    elif ndim == 5:
        return img[:, :, z_indices[:, None], y_indices[:, None], x_indices[None, :]]
    else:
        raise RuntimeError("Symmetric padding of N-D tensors are not supported yet")

def pad(img: Tensor, padding: List[int], fill: int = 0, padding_mode: str = "constant") -> Tensor:
    _assert_image_tensor(img)

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
        raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

    if isinstance(padding, int):
        if torch.jit.is_scripting():
            # This maybe unreachable
            raise ValueError("padding can't be an int while torchscripting, set it as a list [value, ]")
        pad_front = pad_back = pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_front = pad_back = pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 3:
        pad_front = pad_back = padding[0]
        pad_left  = pad_right = padding[1]
        pad_top   = pad_bottom = padding[2]
    else:
        pad_front = padding[0]
        pad_left = padding[1]
        pad_top = padding[2]
        pad_back = padding[3]
        pad_right = padding[4]
        pad_bottom = padding[5]

    p = [pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom]

    if padding_mode == "edge":
        # remap padding_mode str
        padding_mode = "replicate"
    elif padding_mode == "symmetric":
        # route to another implementation
        return _pad_symmetric(img, p)

    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if (padding_mode != "constant") and img.dtype not in (torch.float32, torch.float64):
        # Here we temporary cast input tensor to float
        # until pytorch issue is resolved :
        # https://github.com/pytorch/pytorch/issues/40763
        need_cast = True
        img = img.to(torch.float32)

    img = torch_pad(img, p, mode=padding_mode, value=float(fill))

    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        img = img.to(out_dtype)

    return img

def crop(img: Tensor, front: int, top: int, left: int, depth: int, height: int, width: int) -> Tensor:
    _assert_image_tensor(img)

    return img[..., front:front+depth, top:top + height, left:left + width]

class RandomCrop3D(torch.nn.Module):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        d, w, h = _get_image_size(img)
        td, th, tw = output_size

        if d + 1 < td or h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if d == td and w == tw and h == th:
            return 0, 0, 0, d, h, w

        k = torch.randint(0, d - td + 1, size=(1, )).item()
        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return k, i, j, td, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        img = sample['hr']
        if self.padding is not None:
            img = pad(img, self.padding, self.fill, self.padding_mode)

        depth, width, height = _get_image_size(img)
        # pad the depth if needed
        if self.pad_if_needed and depth < self.size[0]:
            padding = [self.size[0] - width, 0]
            img = pad(img, padding, self.fill, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[2]:
            padding = [self.size[2] - width, 0]
            img = pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[1]:
            padding = [0, self.size[1] - height]
            img = pad(img, padding, self.fill, self.padding_mode)

        k, i, j, d, h, w = self.get_params(img, self.size)

        sample['hr'] = crop(img, k, i, j, d, h, w)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)

class ToTensor:
    def __call__(self, sample):
        for elem in sample.keys():
            temp = sample[elem]
            if isinstance(temp, (list,str)): continue
            if temp.ndim == 4:
                if elem == 'seg':
                    temp = torch.from_numpy(temp)
                else:
                    if isinstance(temp, np.ndarray):
                        temp = torch.from_numpy(temp).subtract(temp.min())
                        temp = temp.div(temp.max())
                    else:
                        temp = temp.subtract(temp.min())
                        temp = temp.div(temp.max())
                sample[elem] = temp.type(torch.FloatTensor)
            else:
                sample[elem] = torch.from_numpy(temp).type(torch.FloatTensor)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ScaleNRotateNShift(object):
    """Scale (zoom-in, zoom-out) and Rotate the image
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), shifts=(-32,32)):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.shifts = shifts

    def __call__(self, sample):
        rot = np.random.uniform(low=self.rots[0], high=self.rots[1])
        sc  = np.random.uniform(low=self.scales[0], high=self.scales[1])
        sh  = np.random.uniform(low=self.shifts[0], high=self.shifts[1])
        for elem in ['img']:
            img = sample[elem]
            h, w = img.shape[-2:]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)
            T = np.float32([[1,0,sh], [0,1,sh]])

            flagval = cv2.INTER_CUBIC
            out = np.zeros(img.shape)
            for i in range(img.shape[1]): 
                t = cv2.warpAffine(np.uint8(img[0,i]), M, (w,h), flags=flagval)
                out[0,i] = cv2.warpAffine(np.uint8(t), T, (w,h))
            del img
            sample[elem] = out
        return sample

    def __str__(self):
        return 'ScaleNRotateNShift:(rot='+str(self.rots)+',scale='+str(self.scales)+' ,shift='+str(self.shifts)+')'

class Transpose(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, tr=(0,2,3,1)):
        self.tr = tr

    def __call__(self, sample):
        if 'img' in sample.keys():
            sample['img'] = np.transpose(sample['img'], self.tr)
        return sample

class ImputeMissingValues(object):
    def __init__(self, method, fold, split):
        import pickle
        self.split = split
        self.fold = fold
        self.mean_vals_before_normalization = {1:[1,1,80.8615,12.6479],2:[1,1,81.4609,12.5957],3:[1,1,81.1156,12.672],4:[1,1,81.1044,12.6044],5:[1,1,80.0928,12.455]}
        self.method = method
        with open("phgen.txt","rb") as f: self.ph = pickle.load(f)
        with open("pxghgen.txt","rb") as f: self.pxgh = pickle.load(f)
        self.eps = 1e-9
        self.fvc_bins = np.array([26., 51.4131795, 56.99145928, 60.04722793, 64.00657197, 66.85635996, 68.90336484, 72., 74.49059669, 77., 79.04399063, 82., 84., 86., 89., 91., 94.09979707, 98.74254553, 104., 110.275, 135.7723577])
        self.dlco_bins = np.array([1.26, 5.00623267494, 6.104, 7.103, 7.774, 8.67, 9.284, 9.789, 10.286000000000001, 10.908999999999999, 11.84, 12.437, 12.904, 13.262, 14.058, 14.885, 15.724, 17.244, 19.092, 20.642999999999997, 61.0])
        self.age_bins = np.array([35.0, 57.0, 62.0, 65.0, 67.0, 69.0, 71.0, 73.0, 75.0, 78.0, 91.0])

    def cont_to_discrete(self,x):
        x = x.squeeze()
        y = np.copy(x)
        for idx, item in enumerate(y):
            if np.isnan(item) or (idx in [1,2,3]):
                continue
            elif idx == 0:
                if item < self.age_bins.min():
                    y[idx] = 0
                elif item >= self.age_bins.max():
                    y[idx] = len(self.age_bins) - 2
                else:
                    y[idx] = np.where(self.age_bins<=item)[0][-1]
                # y[idx] = np.where(self.age_bins<=item)[0][-1]
            elif idx == 4:
                if item < self.fvc_bins.min():
                    y[idx] = 0
                elif item >= self.fvc_bins.max():
                    y[idx] = len(self.fvc_bins) - 2
                else:
                    y[idx] = np.where(self.fvc_bins<=item)[0][-1]
                # y[idx] = np.where(self.fvc_bins<=item)[0][-1]
            else:
                assert idx == 5
                if item < self.dlco_bins.min():
                    y[idx] = 0
                elif item >= self.dlco_bins.max():
                    y[idx] = len(self.dlco_bins) - 2
                else:
                    y[idx] = np.where(self.dlco_bins<=item)[0][-1]
                # assert idx == 5
                # y[idx] = np.where(self.dlco_bins<=item)[0][-1]
        return y

    def discrete_to_cont(self, x, locs):
        mean_vals = []
        for j,k in zip(locs[:-1], locs[1:]): mean_vals.append(np.mean((j,k)))
        return mean_vals[int(x)]

    def condp(self, x, dist_var=None):
        x = x.copy()
        if dist_var is None:
            y = x/x.sum()
        else:
            other_var = [i for i in np.arange(x.ndim) if i not in dist_var]
            y = x.transpose(other_var + dist_var)
            m = {}
            for i,j in zip(range(x.ndim),other_var+dist_var): m[j]=i # remember mapping between axes to perform anti-transpose
            y = y.reshape(*np.array(x.shape)[other_var],-1)
            y /= (y.sum(-1, keepdims=True)+self.eps)
            y = y.reshape(*np.array(x.shape)[other_var], *np.array(x.shape)[dist_var])
            y = y.transpose([m[i] for i in range(y.ndim)]) 
        return y

    def condexp(self, logp, dist_var=None):
        out = np.exp(logp-logp.max())
        return self.condp(out, dist_var)

    def impute_x(self, x, ph, pxgh, k, selection_method):
        t = x.copy()
        t = self.cont_to_discrete(t)
        logpxmissing = np.log(ph).copy()
        obs_idx = np.where(np.isnan(t)==False)[0]
        miss_idx = np.where(np.isnan(t)==True)[0]
        st = {} 
        for c in range(k.max()): 
            st[c] = np.where(t==c)[0]
            for j in st[c]: logpxmissing += np.log(pxgh[j][:,c]+self.eps) # obs data

        combs = np.array(np.meshgrid(*[np.arange(k[m]).tolist() for m in miss_idx])).T.reshape(-1,len(miss_idx))
        # logpxmissing = logpxmissing + np.log(pxgh[miss_idx,:,combs]).sum(1)

        # for j in miss_idx: logpxmissing = logpxmissing[:,None] + np.log(pxgh[j][:,combs]).sum(2)

        ###
        temp = []
        for i,j in enumerate(miss_idx): temp.append(np.log(pxgh[j][:,combs[:,i]]+self.eps))
        temp = np.asarray(temp).sum(0)
        logpxmissing = logpxmissing[:,None] + temp
        ###

        pxmissing = self.condexp(logpxmissing).sum(0) # sum over h
        if selection_method == 'argmax':
            vals = combs[np.argmax(pxmissing,0)]
        elif selection_method == 'mean':
            vals = (pxmissing[:,None]*combs).sum(0)
            vals = np.round(vals).astype(int)
        elif selection_method == 'sample':
            idx = np.random.choice(np.arange(len(combs)), size=1, p=pxmissing)[0]
            vals = combs[idx]
        else:
            assert False

        x[miss_idx] = vals
        # sampling
        # idx = np.random.choice(np.arange(len(dist)), size = 1, p = dist)
        # sample = combs[idx]
        return vals, pxmissing, combs, x

    def __call__(self, sample):
        temp = sample['clinical_data']
        if np.isnan(temp).sum() > 0:
            miss_idx = np.where(np.isnan(temp)==True)[0]
            if self.method == 'mixture':
                if self.split == 'train':
                    temp = self.impute_x(temp, ph=self.ph, pxgh=self.pxgh, k=np.array([10, 2, 3, 2, 20, 20]), selection_method='sample')[-1]
                else:
                    temp = self.impute_x(temp, ph=self.ph, pxgh=self.pxgh, k=np.array([10, 2, 3, 2, 20, 20]), selection_method='argmax')[-1]
                for m in miss_idx:
                    if m in [2,3]:
                        continue
                    elif m == 4:
                        temp[m] = self.discrete_to_cont(temp[m], self.fvc_bins)
                    elif m == 5:
                        temp[m] = self.discrete_to_cont(temp[m], self.dlco_bins)
            elif self.method == 'mean':
                for m in miss_idx:
                    assert m in [2,3,4,5]
                    # after normalization, mean is zero
                    if m in [2,3]:
                        temp[m] = 1
                    else:
                        temp[m] = 0
            elif self.method == 'zeros':
                for m in miss_idx:
                    assert m in [2,3,4,5]
                    # use the mean before norm, so that after norm it becomes zero
                    if m in [2,3]:
                        temp[m] = 0
                    else:
                        temp[m] = self.mean_vals_before_normalization[self.fold][m-2]
        sample['clinical_data'] = temp
        return sample

class Normalize(object):
    def __init__(self, fold):
        self.fold = fold
        self.mean_std_vals = {
            1: {'age': [68.9356,8.0394], 'contemporaneous_fvc_percent': [80.8615,17.6683], 'contemporaneous_dlco': [12.6479,5.9546]},
            2: {'age': [68.4901,8.655], 'contemporaneous_fvc_percent': [81.4609,18.4631], 'contemporaneous_dlco': [12.5957,4.9688]},
            3: {'age': [68.6953,8.7847], 'contemporaneous_fvc_percent': [81.1156,17.8381], 'contemporaneous_dlco': [12.672,5.8646]},
            4: {'age': [68.486,8.4722], 'contemporaneous_fvc_percent': [81.1044,17.7478], 'contemporaneous_dlco': [12.6044,5.9117]},
            5: {'age': [68.3493,8.7852], 'contemporaneous_fvc_percent': [80.0928,18.241], 'contemporaneous_dlco': [12.455,5.5981]}
        }
    def __call__(self, sample):
        temp = sample['clinical_data']
        temp[[0,4,5]] -= np.array([self.mean_std_vals[self.fold][i][0] for i in ['age','contemporaneous_fvc_percent','contemporaneous_dlco']])
        temp[[0,4,5]] /= np.array([self.mean_std_vals[self.fold][i][1] for i in ['age','contemporaneous_fvc_percent','contemporaneous_dlco']])
        sample['clinical_data'] = temp
        return sample