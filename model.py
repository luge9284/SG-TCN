import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import math
import copy
import random
from torch.utils.tensorboard import SummaryWriter
import pdb

class MultiStageModel(nn.Module):
    def __init__(self, num_stages=5, num_layers_F=11,num_layers=10, num_f_maps=64, dim=2048, classes=48):
        super(MultiStageModel, self).__init__()
        self.num_stages = num_stages
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, classes)
        self.stage2 = DDL(num_layers_F, num_f_maps)
        self.stage3 = SingleStageModel(num_layers, num_f_maps, classes, classes)
        self.Semantic_auxiliary1 = Semantic_auxiliary(dim=dim, outdim=classes)
        self.Semantic_auxiliary2 = Semantic_auxiliary(dim=classes, outdim=classes)
        self.conv_in = nn.ModuleList([copy.deepcopy(nn.Conv1d(classes, num_f_maps, 1)) for s in range(num_stages-2)])
        self.conv_out = nn.ModuleList([copy.deepcopy(nn.Conv1d(num_f_maps, classes, 1)) for s in range(num_stages-2)])
        self.Pace_pre = Pace_pre(dim=classes, classes=4)
        self.atten1 = Atten_Fusion_Conv(classes)
        self.atten2 = Atten_Fusion_Conv(classes)
        self.atten3 = Atten_Fusion_Conv(classes)

    def forward(self, x, mask, pace):
        tcn_out = self.stage1(x, mask)
        semantic_out = self.Semantic_auxiliary1(x, mask)
        out = self.atten1(semantic_out, tcn_out)
        outputs = out.unsqueeze(0)
        pace_out = self.Pace_pre(F.softmax(out, dim=1) * mask[:, 0:1, :], mask, pace)
        pace_outs = pace_out.unsqueeze(0)


        input = self.conv_in[0](F.softmax(out, dim=1))
        tcn_out = self.stage2(input * mask[:, 0:1, :], mask)
        tcn_out = self.conv_out[0](tcn_out) * mask[:, 0:1, :]
        semantic_out = self.Semantic_auxiliary2(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
        out = self.atten2(semantic_out, tcn_out)
        outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        pace_out = self.Pace_pre(F.softmax(out, dim=1) * mask[:, 0:1, :], mask, pace)
        pace_outs = torch.cat((pace_outs, pace_out.unsqueeze(0)), dim=0)

        input = self.conv_in[1](F.softmax(out, dim=1))
        tcn_out = self.stage2(input * mask[:, 0:1, :], mask)
        tcn_out = self.conv_out[1](tcn_out) * mask[:, 0:1, :]
        semantic_out = self.Semantic_auxiliary2(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
        out = self.atten3(semantic_out, tcn_out)
        outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        pace_out = self.Pace_pre(F.softmax(out, dim=1) * mask[:, 0:1, :], mask, pace)
        pace_outs = torch.cat((pace_outs, pace_out.unsqueeze(0)), dim=0)

        input = self.conv_in[2](F.softmax(out, dim=1))
        out = self.stage2(input * mask[:, 0:1, :], mask)
        out = self.conv_out[2](out) * mask[:, 0:1, :]
        outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        pace_out = self.Pace_pre(F.softmax(out, dim=1) * mask[:, 0:1, :], mask, pace)
        pace_outs = torch.cat((pace_outs, pace_out.unsqueeze(0)), dim=0)


        out = self.stage3(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
        outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        pace_out = self.Pace_pre(F.softmax(out, dim=1) * mask[:, 0:1, :], mask, pace)
        pace_outs = torch.cat((pace_outs, pace_out.unsqueeze(0)), dim=0)

        return outputs, pace_outs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        out = x + out
        return out * mask[:, 0:1, :]


class DDL(nn.Module):
    def __init__(self, num_layers_F, num_f_maps):           #num_layers_PG=11   num_f_maps=64
        super(DDL, self).__init__()
        self.num_layers = num_layers_F
        self.conv_dilated_1 = nn.ModuleList((nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers_F - 1 - i),
                                                       dilation=2 ** (num_layers_F - 1 - i)) for i in
                                             range(num_layers_F)))
        self.conv_dilated_2 = nn.ModuleList(
            (nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i) for i in range(num_layers_F)))
        self.conv_fusion = nn.ModuleList((nn.Conv1d(2 * num_f_maps, num_f_maps, 1) for i in range(num_layers_F)))
        self.dropout = nn.Dropout(0.5)

    def forward(self, f, mask):
        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
            f = f * mask[:, 0:1, :]

        return f


class Semantic_auxiliary(nn.Module):
    def __init__(self, dim=2048, outdim=64):
        super(Semantic_auxiliary, self).__init__()
        self.conv1_1 = nn.Conv1d(dim, 64, 3, padding=1)
        self.conv1_2 = nn.Conv1d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv1d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv1d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv1d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv1d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv1d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv1d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv1d(512, 512, 3, padding=1)

        self.conv4_3d = nn.Conv1d(1024, 512, 3, padding=1)
        self.conv4_2d = nn.Conv1d(512, 512, 3, padding=1)
        self.conv4_1d = nn.Conv1d(512, 256, 3, padding=1)

        self.conv3_3d = nn.Conv1d(512, 256, 3, padding=1)
        self.conv3_2d = nn.Conv1d(256, 256, 3, padding=1)
        self.conv3_1d = nn.Conv1d(256, 128, 3, padding=1)

        self.conv2_2d = nn.Conv1d(256, 128, 3, padding=1)
        self.conv2_1d = nn.Conv1d(128, 64, 3, padding=1)

        self.conv1_2d = nn.Conv1d(128, 64, 3, padding=1)
        self.conv1_1d = nn.Conv1d(64, outdim, 3, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x, mask):
        t = x.shape[2]
        conv1_1 = self.dropout(self.relu(self.conv1_1(x)) * mask[:, 0:1, :])
        conv1_2 = self.dropout(self.relu(self.conv1_2(conv1_1)) * mask[:, 0:1, :])
        pool1 = self.maxpool(conv1_2) * mask[:, 0:1, ::2]

        conv2_1 = self.dropout(self.relu(self.conv2_1(pool1)) * mask[:, 0:1, ::2])
        conv2_2 = self.dropout(self.relu(self.conv2_2(conv2_1)) * mask[:, 0:1, ::2])
        pool2 = self.maxpool(conv2_2) * mask[:, 0:1, ::4]

        conv3_1 = self.dropout(self.relu(self.conv3_1(pool2)) * mask[:, 0:1, ::4])
        conv3_2 = self.dropout(self.relu(self.conv3_2(conv3_1)) * mask[:, 0:1, ::4])
        conv3_3 = self.dropout(self.relu(self.conv3_3(conv3_2)) * mask[:, 0:1, ::4])
        pool3 = self.maxpool(conv3_3) * mask[:, 0:1, ::8]

        conv4_1 = self.dropout(self.relu(self.conv4_1(pool3)) * mask[:, 0:1, ::8])
        conv4_2 = self.dropout(self.relu(self.conv4_2(conv4_1)) * mask[:, 0:1, ::8])
        conv4_3 = self.dropout(self.relu(self.conv4_3(conv4_2)) * mask[:, 0:1, ::8])
        pool4 = self.maxpool(conv4_3) * mask[:, 0:1, ::16]

        conv5_1 = self.dropout(self.relu(self.conv5_1(pool4)) * mask[:, 0:1, ::16])
        conv5_2 = self.dropout(self.relu(self.conv5_2(conv5_1)) * mask[:, 0:1, ::16])
        conv5_3 = self.dropout(self.relu(self.conv5_3(conv5_2)) * mask[:, 0:1, ::16])

        c6_1 = torch.cat(
            (F.interpolate(conv5_3, math.ceil(t / 8), mode='linear', align_corners=True) * mask[:, 0:1, ::8], conv4_3),
            dim=1)
        conv4_3d = self.dropout(self.relu(self.conv4_3d(c6_1)) * mask[:, 0:1, ::8])
        conv4_2d = self.dropout(self.relu(self.conv4_2d(conv4_3d)) * mask[:, 0:1, ::8])
        conv4_1d = self.dropout(self.relu(self.conv4_1d(conv4_2d)) * mask[:, 0:1, ::8])

        c7_1 = torch.cat(
            (F.interpolate(conv4_1d, math.ceil(t / 4), mode='linear', align_corners=True) * mask[:, 0:1, ::4], conv3_3),
            dim=1)
        conv3_3d = self.dropout(self.relu(self.conv3_3d(c7_1)) * mask[:, 0:1, ::4])
        conv3_2d = self.dropout(self.relu(self.conv3_2d(conv3_3d)) * mask[:, 0:1, ::4])
        conv3_1d = self.dropout(self.relu(self.conv3_1d(conv3_2d)) * mask[:, 0:1, ::4])

        c8_1 = torch.cat(
            (F.interpolate(conv3_1d, math.ceil(t / 2), mode='linear', align_corners=True) * mask[:, 0:1, ::2], conv2_2),
            dim=1)
        conv2_2d = self.dropout(self.relu(self.conv2_2d(c8_1)) * mask[:, 0:1, ::2])
        conv2_1d = self.dropout(self.relu(self.conv2_1d(conv2_2d)) * mask[:, 0:1, ::2])

        c9_1 = torch.cat((F.interpolate(conv2_1d, t, mode='linear', align_corners=True) * mask[:, 0:1, :], conv1_2),
                         dim=1)
        conv1_2d = self.dropout(self.relu(self.conv1_2d(c9_1)) * mask[:, 0:1, :])
        output = self.dropout(self.relu(self.conv1_1d(conv1_2d)) * mask[:, 0:1, :])

        return output


class Pace_pre(nn.Module):
    def __init__(self, dim=64, classes=4):
        super(Pace_pre, self).__init__()
        self.dim = dim
        self.classes = classes
        self.conv1_1 = nn.Conv1d(self.dim, 64, 3, padding=1)
        self.conv1_2 = nn.Conv1d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv1d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv1d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv1d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv1d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv1d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv1d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv1d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv1d(512, 512, 3, padding=1)
        
        self.fc = nn.Linear(512, self.classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, mask, pace):  # x(b, c, t)
        pace_out = torch.zeros((x.shape[0], self.classes))  # (b, 4)
        for i in range(x.shape[0]):
            s_x = x[i, :, ::int(pace[i]) + 1]
            s_mask = mask[i, 0, ::int(pace[i]) + 1]
            s_x = torch.unsqueeze(s_x, dim=0)  # (b, 4, t')
            s_mask = torch.unsqueeze(s_mask, dim=0)
            s_mask = s_mask.repeat(1, self.dim, 1)
            s_x = s_x[s_mask.bool()].view(1, self.dim, -1)

            conv1_1 = self.dropout(self.relu(self.conv1_1(s_x)))
            conv1_2 = self.dropout(self.relu(self.conv1_2(conv1_1)))
            pool1 = self.maxpool(conv1_2)

            conv2_1 = self.dropout(self.relu(self.conv2_1(pool1)))
            conv2_2 = self.dropout(self.relu(self.conv2_2(conv2_1)))
            pool2 = self.maxpool(conv2_2)

            conv3_1 = self.dropout(self.relu(self.conv3_1(pool2)))
            conv3_2 = self.dropout(self.relu(self.conv3_2(conv3_1)))
            conv3_3 = self.dropout(self.relu(self.conv3_3(conv3_2)))
            pool3 = self.maxpool(conv3_3)

            conv4_1 = self.dropout(self.relu(self.conv4_1(pool3)))
            conv4_2 = self.dropout(self.relu(self.conv4_2(conv4_1)))
            conv4_3 = self.dropout(self.relu(self.conv4_3(conv4_2)))
            pool4 = self.maxpool(conv4_3)

            conv5_1 = self.dropout(self.relu(self.conv5_1(pool4)))
            conv5_2 = self.dropout(self.relu(self.conv5_2(conv5_1)))
            conv5_3 = self.dropout(self.relu(self.conv5_3(conv5_2)))

            out = torch.mean(conv5_3, dim=2)

            out = self.dropout(self.fc(out))

            out = torch.squeeze(out)
            pace_out[i, :] = out

        return pace_out


class Atten_Layer(nn.Module):
    def __init__(self, channel):
        super(Atten_Layer, self).__init__()
        self.sc = channel
        self.tc = channel
        rc = self.sc // 2
        self.fc1 = nn.Linear(self.sc, rc)
        self.fc2 = nn.Linear(self.tc, rc)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, Semantic_features, TCN_features):
        batch = Semantic_features.size(0)
        Semantic_features = Semantic_features.transpose(1, 2).contiguous().view(-1, self.sc) #BCN->BNC->(BN)C
        TCN_features = TCN_features.transpose(1, 2).contiguous().view(-1, self.tc) #BCN->BNC->(BN)C'
        rs = self.fc1(Semantic_features)
        rt = self.fc2(TCN_features)
        att = torch.sigmoid(self.fc3(torch.tanh(rs + rt))) #BNx1
        out = Semantic_features * att
        out = out.view(batch, Semantic_features.size(1), -1)

        return out


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inchannel):
        super(Atten_Fusion_Conv, self).__init__()
        self.IA_Layer = Atten_Layer(inchannel)

    def forward(self, Semantic_features, TCN_features):
        Semantic_features = self.IA_Layer(Semantic_features, TCN_features)
        fusion_features = Semantic_features + TCN_features

        return fusion_features


class Trainer:
    def __init__(self, num_classes):
        self.model = MultiStageModel(classes=num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_p = nn.CrossEntropyLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, log_path, bz, batch_gen, vid_list_file, features_path, sample_rate, actions_dict, gt_path, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)

        logger = SummaryWriter(log_dir="./tb_log")

        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        log_file = open(log_path, 'w')
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            item = 0

            num_iter = 0
            num_steps_per_update = 2
            step_loss = 0

            test_total = 0
            test_correct = 0
            
            optimizer.zero_grad()

            while batch_gen.has_next():
                num_iter += 1
                batch_input, batch_target, mask, batch_pace = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask, batch_pace = batch_input.to(device), batch_target.to(device), mask.to(device), batch_pace.to(device)

                predictions, pace_outs = self.model(batch_input, mask, batch_pace)
                pace_loss = 0
                ce_loss = 0
                mse_loss = 0

                for pace_out in pace_outs:
                    pace_loss += 0.2 * self.ce_p(F.softmax(pace_out, dim=1).contiguous().view(-1, 4).to(device), batch_pace.view(-1).long())

                for p in predictions:
                    ce_loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    mse_loss += 0.15 * torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16) * mask[:, :, 1:])

                step_loss += pace_loss + ce_loss + mse_loss

                if not batch_gen.has_next():
                    item += 1
                    step_loss.backward()
                    optimizer.step()
                    epoch_loss += step_loss.item()
                    optimizer.zero_grad()

                    _, predicted = torch.max(predictions[-1].data, 1)
                    correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                    total += torch.sum(mask[:, 0, :]).item()
                    print("[epoch %d]:item = %d   loss = %f,   acc = %f" % (epoch + 1, item, epoch_loss / (bz * (item-1) * num_steps_per_update + bz * num_iter), float(correct) / total))
                    step_loss = 0
                    num_iter = 0

                if num_iter == num_steps_per_update:
                    item += 1
                    step_loss.backward()
                    optimizer.step()
                    epoch_loss += step_loss.item()
                    optimizer.zero_grad()

                    _, predicted = torch.max(predictions[-1].data, 1)
                    correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                    total += torch.sum(mask[:, 0, :]).item()
                    print("[epoch %d]:item = %d   loss = %f,   acc = %f" % (epoch + 1, item, epoch_loss / (bz * item * num_steps_per_update), float(correct) / total))
                    step_loss = 0
                    num_iter = 0

            logger.add_scalar("train loss", epoch_loss, epoch + 1)

            with torch.no_grad():

                for vid in list_of_vids:
                    features = np.load(features_path + vid.split('.')[0] + '.npy')
                    gt_file = gt_path + vid
                    with open(gt_file, 'r') as f:
                        content = f.read()
                        f.close()
                    gt_content = content.split('\n')[0:-1]
                    features = features[:, ::sample_rate]
                    input_x = torch.tensor(features, dtype=torch.float)
                    input_x.unsqueeze_(0)
                    input_x = input_x.to(device)
                    pace = torch.ones(1)
                    predictions, _ = self.model(input_x, torch.ones(input_x.size(), device=device), pace)
                    _, predicted = torch.max(predictions[-1].data, 1)
                    predicted = predicted.squeeze()
                    recognition = []
                    for i in range(len(predicted)):
                        recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                        list(actions_dict.values()).index(
                                                                            predicted[i].item())]] * sample_rate))

                    for i in range(len(gt_content)):
                        test_total += 1
                        if gt_content[i] == recognition[i]:
                            test_correct += 1

                test_acc = 100 * float(test_correct) / test_total

            logger.add_scalar("test accuary", test_acc, epoch + 1)

            batch_gen.reset()
            random.shuffle(batch_gen.list_of_examples)
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), float(correct) / total))
            log_file.write("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), float(correct) / total))
            log_file.write('\n')


    def predict(self, model_dir, results_dir, features_path, vid_list_file, predict_load_epoch, actions_dict, device,
                sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(predict_load_epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                pace = torch.ones(1)
                predictions, _ = self.model(input_x, torch.ones(input_x.size(), device=device), pace)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))

                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

