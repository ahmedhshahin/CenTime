import subprocess
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int, default=1)
parser.add_argument('-hours', type=int, default=24)
parser.add_argument('-tmem', type=int, default=16)

parser.add_argument('-optimizer_type', type=str, default='adam')
parser.add_argument('-n_epochs', type=int)
parser.add_argument('-lr_policy', type=str, default='none')
parser.add_argument('-model', type=str)
parser.add_argument('-base_filters', default=4, type=int)
parser.add_argument('-normalization', type=str, default='bn')
parser.add_argument('-clinical_dim', default=7, type=int)
parser.add_argument('-lr', type=float)
parser.add_argument('-weight_decay', default=0, type=float)
parser.add_argument('-factor', type=float, default=0.1)
parser.add_argument('-augment', type=int)
parser.add_argument('-batch_size', type=int)
parser.add_argument('-clinical_exp', default=0, type=int)
parser.add_argument('-fold_start', default=1, type=int)
parser.add_argument('-fold_end', default=6, type=int)
parser.add_argument('-n', default=None, type=int)
parser.add_argument('-patience', default=30, type=int)
parser.add_argument('-disc', type=str, default=None)
parser.add_argument('-impute', type=str, default=None)
args = parser.parse_args()


if 'img_only' in args.model:
    exp_name = 'ImgOnly'
elif 'daft' in args.model:
    exp_name = 'Daft'
elif 'concat' in args.model:
    exp_name = 'Concat'
exp_name += 'Lr{}_{}'.format(str(int(np.log10(args.lr))), args.base_filters)
if args.disc is not None: exp_name = args.disc + exp_name

txt = []
txt.append('#$ -l gpu={}'.format(bool(args.gpu)))
txt.append('#$ -l h_rt={}:00:0'.format(args.hours))
txt.append('#$ -l tmem={}G'.format(args.tmem))
txt.append('#$ -S /bin/bash')
txt.append('#$ -j y')
txt.append('#$ -N {}'.format(exp_name))
txt.append('hostname')
txt.append('date')

dir_path = os.path.dirname(os.path.realpath(__file__))
# txt.append('#$ -o {}/{}/{}'.format(dir_path,'logs',exp_name))
if 'AW' in args.disc:
    txt.append('#$ -o {}/{}/{}'.format(dir_path,'logs/aw',exp_name))
elif 'AN' in args.disc:
    txt.append('#$ -o {}/{}/{}'.format(dir_path,'logs/an',exp_name))
else:
    txt.append('#$ -o {}/{}/{}'.format(dir_path,'logs',exp_name))

txt.append('#$ -e {}/Errors'.format(dir_path))
txt.append('#$ -wd {}'.format(dir_path))
if args.gpu:
    txt.append('if (( $CUDA_VISIBLE_DEVICES > -1 ))')
    txt.append('then')
txt.append('conda activate ahmedenv')
txt.append('export CUBLAS_WORKSPACE_CONFIG=:4096:8')

if args.n is None:
    txt.append('python -u {}/train_cv.py --lr {} --augment {} --n_epochs {} --exp_name {} --model {} --batch_size {} --normalization {} --factor {} --lr_policy {} --optimizer_type {} --base_filters {} --weight_decay {} --clinical_dim {} --patience {} --fold_start {} --fold_end {} --clinical_exp {} --impute {}'.format(dir_path, args.lr,args.augment,args.n_epochs,exp_name,args.model,args.batch_size,args.normalization,args.factor,args.lr_policy, args.optimizer_type, args.base_filters, args.weight_decay, args.clinical_dim, args.patience, args.fold_start, args.fold_end, args.clinical_exp, args.impute))
else:
    txt.append('python -u {}/train_cv.py --lr {} --augment {} --n_epochs {} --exp_name {} --model {} --batch_size {} --normalization {} --factor {} --lr_policy {} --optimizer_type {} --base_filters {} --weight_decay {} --clinical_dim {} --patience {} --fold_start {} --fold_end {} --n {}'.format(dir_path, args.lr,args.augment,args.n_epochs,exp_name,args.model,args.batch_size,args.normalization,args.factor,args.lr_policy, args.optimizer_type, args.base_filters, args.weight_decay, args.clinical_dim, args.patience, args.fold_start, args.fold_end, args.n))


if args.gpu:
    txt.append('fi')


file = open('autosubmit.sh','w')
file.writelines('\n'.join(txt))
file.close()

comm = 'qsub autosubmit.sh'
process = subprocess.Popen(comm.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
try:
    print(error.decode())
except:
    print(output.decode())