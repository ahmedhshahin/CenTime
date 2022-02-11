import os
import numpy as np

import torch

# Env
from options import parse_args
# from train_test import train_test
import train_test, train_test_clinical

from torch.utils.tensorboard import SummaryWriter

### 1. Initializes parser and device
opt = parse_args()
if opt.clinical_exp == 0:
	# from train_test import train_test
	from train_test import train_test
elif opt.clinical_exp == 1:
	from train_test_clinical import train_test
elif opt.clinical_exp == 2:
	from train_test_new import train_test
# assert False

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)
if torch.cuda.device_count() > 1: print("Using {} GPUs".format(torch.cuda.device_count()))
if not os.path.exists(opt.exps_dir): os.makedirs(opt.exps_dir)
if not os.path.exists(os.path.join(opt.exps_dir, opt.exp_name)): os.makedirs(os.path.join(opt.exps_dir, opt.exp_name))

results = []
### 3. Sets-Up Main Loop
for fold in range(int(opt.fold_start), int(opt.fold_end)):
	print("*******************************************")
	print("************** SPLIT (%d/%d) **************" % (fold, 5))
	print("*******************************************")
	writer = SummaryWriter(log_dir=os.path.join(opt.exps_dir, opt.exp_name))
	### 3.1 Trains Model
	res = train_test(opt, fold, device, writer)
	results.append(res)
	print()
results = np.array(results)
print('Split Results:', results)
print("Average: {}, STD: {}".format(results.mean(), results.std()))