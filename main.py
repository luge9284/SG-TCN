import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
import pdb



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 808808808
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

print(torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')

args = parser.parse_args()


features_dim = 2048
bz = 1
lr = 0.0005
num_epochs = 100

predict_load_epoch = 100

sample_rate = 1

if args.dataset == "50salads":
    sample_rate = 2

content = 'your/data/path'
vid_list_file = content + args.dataset + "/splits/train.split" + args.split + ".bundle"
vid_list_file_tst = content + args.dataset + "/splits/test.split" + args.split + ".bundle"
features_path = content + args.dataset + "/features/"
gt_path = content + args.dataset + "/groundTruth/"

mapping_file = content + args.dataset + "/mapping.txt"
model_dir = "./models/"+args.dataset+"/split_"+args.split
results_dir = "./results/"+args.dataset+"/split_"+args.split
log_dir = "./log/"+args.dataset+"/split_"+args.split

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

log_path = log_dir + "/" + args.dataset+"-split_"+args.split + ".txt"

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

trainer = Trainer(num_classes)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, log_path, bz, batch_gen, vid_list_file_tst, features_path, sample_rate, actions_dict, gt_path, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, predict_load_epoch, actions_dict, device, sample_rate)
