import os
import logging
from abc import abstractmethod
import json
import numpy as np
import time
import torch
import pandas as pd
from scipy import sparse
from numpy import inf
from tqdm import tqdm
from tensorboardX import SummaryWriter

METRICS = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'CIDEr', 'ROUGE_L']


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args
        # tensorboard 记录参数和结果
        self.writer = SummaryWriter(args.save_dir)
        self.print_args2tensorbord()

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            # 如果使用多个GPU训练，则将model模型对象转换为一个支持多GPU并行训练的模型对象
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs # 训练的epoch个数
        self.save_period = self.args.save_period # 保存周期

        self.mnt_mode = args.monitor_mode
        # 验证集和测试集的评价指标
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        # 如果 self.mnt_mode 的值为 'max'，则模型的性能指标越大越好，因此将 self.mnt_best 初始化为负无穷大，以便在后续的训练过程中找到更大的性能指标
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf # 用于记录最佳性能指标的值
        self.early_stop = getattr(self.args, 'early_stop', inf) # 控制在连续多少个epoch中性能指标没有得到改善时停止训练

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            # 是否从现有检查点恢复训练
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}
        # print("BaseTrainer init completed")
        logging.info(f"BaseTrainer init completed")

    @abstractmethod
    def _train_epoch(self, epoch):
        # _train_epoch是BaseTrainer基类中定义的抽象类，在子类中必须要实现，否则会报错
        raise NotImplementedError

    def train(self):
        not_improved_count = 0 # 记录模型在多少个epoch中性能指标没有得到改善，如果得到改善该变量会重置为0，用于判断模型训练什么时候可以停止
        for epoch in range(self.start_epoch, self.epochs + 1):
            try:
                logging.info(f'==>> Model lr: {self.optimizer.param_groups[1]["lr"]:.7}, '
                             f'Visual Encoder lr: {self.optimizer.param_groups[0]["lr"]:.7}')
                # result是当前epoch的训练日志，包含训练损失、验证和测试的指标结果
                result = self._train_epoch(epoch)

                # save logged informations into log dict
                log = {'epoch': epoch}
                log.update(result)
                # 更新验证集和测试集的最佳记录
                self._record_best(log)

                # 打印日志信息，包括：
                # ①当前的训练轮次(epoch)以及检查点目录(checkpoint_dir)
                # ②将验证集和测试集结果加入到日志中，并在控制台打印出来
                # ③添加验证集和测试集的评价指标及其对应的周期数到tensorboard中
                self._print_epoch(log)

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                improved = False
                if self.mnt_mode != 'off':
                    # self.mnt_mode只有两种选择，不可能是'off'
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        # 先判断是否比之前更好，判断的依据是验证集的BLEU_4指标是否比之前更大
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        logging.error(
                            "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                                self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        # 如果更好了，则更新最佳记录
                        # 更新的时候，还是将验证集的指标值给了mnt_best
                        # 个人理解，与self._record_best(log)的区别是：这里只记录了验证集的BLEU_4指标值，
                        # 而_record_best(log)记录了验证集和测试集的指标值
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0

                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        # 没有改进就停止训练
                        logging.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        break # 跳出epoch，即停止迭代

                if epoch % self.save_period == 0:
                    # 达到设定的保存周期时，保存当前epoch的检查点，肯定会保存当前epoch的检查点(要么是interrupt要么是current)
                    # 如果这一个epoch的验证集指标更好，则还会保存截至目前的最佳检查点
                    # 这里设置的是save_period=1，即每个epoch都会保存一次检查点
                    self._save_checkpoint(epoch, save_best=improved)
            except KeyboardInterrupt:
                logging.info('=> User Stop!')
                self._save_checkpoint(epoch, save_best=False, interrupt=True)
                logging.info('Saved checkpint!')
                if epoch > 1:
                    self._print_best()
                    self._print_best_to_file()
                return
        # 所有的epoch结束后，保存最后的检查点
        self._print_best()
        # 并保存到文件
        self._print_best_to_file()

    def print_args2tensorbord(self):
        # 遍历了 self.args 中的所有参数，并将它们的名称和值以文本的形式添加到 TensorBoard 中
        for k, v in vars(self.args).items():
            self.writer.add_text(k, str(v))

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        for split in ['val', 'test']:
            self.best_recorder[split]['version'] = f'V{self.args.version}'
            self.best_recorder[split]['visual_extractor'] = self.args.visual_extractor
            self.best_recorder[split]['time'] = crt_time
            self.best_recorder[split]['seed'] = self.args.seed
            self.best_recorder[split]['best_model_from'] = 'val'
            self.best_recorder[split]['lr'] = self.args.lr_ed

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        # n_gpu_use是使用的GPU的数量，默认为1；不是要使用的GPU的编号
        n_gpu = torch.cuda.device_count() # 获取设备GPU的数量
        if n_gpu_use > 0 and n_gpu == 0:
            # 不存在GPU时，打印警告信息，并将n_gpu_use设置为0，之后就使用CPU
            logging.info("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            # 所选择的GPU超过GPU数量时，打印警告信息，并将n_gpu_use设置为n_gpu，即使用最后一个GPU
            logging.info(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu') # 指定 PyTorch 张量要被分配到第一个 GPU 上
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, interrupt=False):
        # 保存检查点，包括两种情况：一个是保存由于临时打断时的当前检查点，另一个是保存当前epoch的最佳的检查点
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        if interrupt:
            filename = os.path.join(self.checkpoint_dir, 'interrupt_checkpoint.pth')
        else:
            filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        logging.debug("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            logging.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        logging.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        logging.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        # 最佳纪录best_recorder中记录的指标是BLEU_4指标的值
        # improved_val是一个逻辑值
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            # 如果improved_val为真，则需要更新验证集的最佳的记录
            self.best_recorder['val'].update(log) # 更新之后最佳记录中就不止有BLEU_4指标的值了
            self.writer.add_text(f'best_BELU4_byVal', str(log["test_BLEU_4"]), log["epoch"]) # 同时将BLEU_4指标的值写入tensorboard中，便于后续可视化等

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)
            # self.writer.add_text(f'best_val_BELU4', str(log["val_BLEU_4"]), log["epoch"])
            self.writer.add_text(f'best_BELU4_byTest', str(log["test_BLEU_4"]), log["epoch"])

    def _print_best(self):
        # 打印最佳的验证集和测试集的结果，同时写入到日志中
        logging.info('\n' + '*' * 20 + 'Best results' + '*' * 20)
        logging.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        self._prin_metrics(self.best_recorder['val'], summary=True)

        logging.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        self._prin_metrics(self.best_recorder['test'], summary=True)

        # For Record
        print(self.checkpoint_dir)
        vlog, tlog = self.best_recorder['val'], self.best_recorder['test']
        if 'epoch' in vlog:
            print(f'Val  set: Epoch: {vlog["epoch"]} | ' + 'loss: {:.4} | '.format(vlog["train_loss"]) + ' | '.join(
                ['{}: {:.4}'.format(m, vlog['test_' + m]) for m in METRICS]))
            print(f'Test Set: Epoch: {tlog["epoch"]} | ' + 'loss: {:.4} | '.format(tlog["train_loss"]) + ' | '.join(
                ['{}: {:.4}'.format(m, tlog['test_' + m]) for m in METRICS]))

            print(','.join(['{:.4}'.format(vlog['test_' + m]) for m in METRICS]) + f',E={vlog["epoch"]}'
                  + f'|TE={tlog["epoch"]} B4={tlog["test_BLEU_4"]:.4}')

    def _prin_metrics(self, log, summary=False):
        # 将验证集和测试集结果加入到日志中，并在控制台打印出来
        if 'epoch' not in log:
            # log里面一定会有epoch这个key，所以这句就不会执行
            logging.info("===>> There are not Best Results during this time running!")
            return
        logging.info(
            f'VAL ||| Epoch: {log["epoch"]}|||' + 'train_loss: {:.4}||| '.format(log["train_loss"]) + ' |||'.join(
                ['{}: {:.4}'.format(m, log['val_' + m]) for m in METRICS]))
        logging.info(
            f'TEST || Epoch: {log["epoch"]}|||' + 'train_loss: {:.4}||| '.format(log["train_loss"]) + ' |||'.join(
                ['{}: {:.4}'.format(m, log['test_' + m]) for m in METRICS]))

        if not summary:
            # 一定会执行这里
            if isinstance(log['epoch'], str):
                # log['epoch']是字符串，则说明是从检查点中恢复训练的，此时需要计算epoch的值
                # 但我们这里一般不会从检查点中恢复训练，所以这里也不会执行
                epoch_split = log['epoch'].split('-')
                e = int(epoch_split[0])
                if len(epoch_split) > 1:
                    it = int(epoch_split[1])
                    epoch = len(self.train_dataloader) * e + it
                else:
                    epoch = len(self.train_dataloader) * e
            else:
                # epoch是训练的代数，因为一个epoch会将所有的训练数据都训练一遍
                # 因此，int(log['epoch']) * len(self.train_dataloader)就可以表示周期编号了
                epoch = int(log['epoch']) * len(self.train_dataloader)

            for m in METRICS:
                # METRICS是几个评价指标的名称，包括BLEU_1、BLEU_2、BLEU_3、BLEU_4、CIDEr和ROUGE_L
                # add_scalar:这是SummaryWriter对象的一个方法，用于向tensorborad中添加标量数据，
                # 这里添加了验证集和测试集的评价指标及其对应的周期数
                self.writer.add_scalar(f'val/{m}', log["val_" + m], epoch)
                self.writer.add_scalar(f'test/{m}', log["test_" + m], epoch)
                # 把训练的损失加入到tensorboard中
                self.writer.add_scalar(f'train/loss', log["train_loss"], epoch)

    def _output_generation(self, predictions, gts, idxs, epoch, iters=0, split='val'):
        # 将模型预测结果、真实结果、迭代信息和数据集划分信息保存到一个json文件中
        # from nltk.translate.bleu_score import sentence_bleu
        output = list()
        for idx, pre, gt in zip(idxs, predictions, gts):
            # score = sentence_bleu([gt.split()], pre.split())
            output.append({'filename': idx, 'prediction': pre, 'ground_truth': gt})

        # output = sorted(output, key=lambda x: x['bleu4'], reverse=True)
        json_file = f'Enc2Dec-{epoch}_{iters}_{split}_generated.json'
        output_filename = os.path.join(self.checkpoint_dir, json_file)
        with open(output_filename, 'w') as f:
            json.dump(output, f, ensure_ascii=False)

    def _print_epoch(self, log):
        # 记录当前的训练轮次(epoch)以及检查点目录(checkpoint_dir)
        logging.info(f"Epoch [{log['epoch']}/{self.epochs}] - {self.checkpoint_dir}")
        self._prin_metrics(log)


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        # print("Trainer init completed")
        logging.info(f"Trainer init completed")

    def _train_epoch(self, epoch):
        # 是基类的_train_epoch抽象方法在子类中的实现

        train_loss = 0
        self.model.train() # 用于将模型设置为训练模式
        # tqdm库，它是一个用于在循环中显示进度条的工具
        # 与 self.train_dataloader 数据加载器关联起来。ncols=80 参数指定了进度条的宽度为80个字符，这样可以实时地查看训练数据的加载进度
        t = tqdm(self.train_dataloader, ncols=80)
        for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(t):
            # 由于target和target_mask的列数与最大序列长度有关，因此在每个batch中，它们的列数可能不同
            # enumerate()函数会返回一个包含索引和元素值的元组，其中索引从0开始，依次递增
            images, reports_ids, reports_masks, labels = images.to(self.device), reports_ids.to(self.device), \
                                                         reports_masks.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad() # 将优化器中的梯度清零
            # 进行模型的计算，包括：视觉特征的提取、文本特征的提取、编码器解码器的计算(即由图像到文本生成的过程)
            outputs = self.model(images, reports_ids, labels, mode='train')
            # outputs[0]就是最终解码器的结果，用(outputs[0],)变成元组是方便和(reports_ids, reports_masks, labels)进行元组的加法；
            # 传入的labels是真实的疾病标签；outputs[1:]是一个元组，包含了compute_loss函数其余的参数(具体可以看笔记记录)
            # 最后传入了self.args，即训练过程中的参数
            loss = self.criterion(*((outputs[0],) + (reports_ids, reports_masks, labels) + outputs[1:] + (self.args,)))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.grad_clip) # 梯度裁剪，防止梯度爆炸
            # 每一个批次都更新一次参数
            self.optimizer.step()

            t.set_description(f'train loss:{loss.item():.3}')
            if self.args.test_steps > 0 and epoch > 1 and (batch_idx + 1) % self.args.test_steps == 0:
                # 由于self.args.test_steps默认为0，所以永远判断失败，不会执行下面的语句
                # self.test_step(epoch, batch_idx + 1)
                # self.model.train()
                self.model.eval()
        log = {'train_loss': train_loss / len(self.train_dataloader)} # 计算平均的训练损失

        ilog = self._test_step(epoch, 0, 'val')
        # 将验证集的评价指标添加到log字典中
        log.update(**ilog)

        # 测试集的计算过程和验证集的计算过程相同
        ilog = self._test_step(epoch, 0, 'test')
        log.update(**ilog)

        # 更新学习率，然后这个epoch结束
        self.lr_scheduler.step()

        return log

    def _test_step(self, epoch, iters=0, mode='test'):
        ilog = {}
        self.model.eval()
        data_loader = self.val_dataloader if mode == 'val' else self.test_dataloader
        with torch.no_grad():
            # 在验证过程中，只需要计算模型的输出结果，而不需要计算梯度
            # torch.no_grad()是一个上下文管理器，用于禁用梯度计算
            # 因此，with torch.no_grad():会在代码块中禁用梯度计算，但代码块外的梯度计算仍然有效

            # val_res存放的是模型预测的结果，val_gts存放的是真实的序列，val_idxs存放的是图像的id
            val_gts, val_res, val_idxs = [], [], []
            t = tqdm(data_loader, ncols=80)
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(t):
                images, reports_ids, reports_masks, labels = images.to(self.device), reports_ids.to(self.device), \
                                                             reports_masks.to(self.device), labels.to(self.device)
                outputs = self.model(images, mode='sample') # 进行模型计算
                # 将预测结果转化成文本
                reports = self.model.tokenizer.decode_batch(outputs[0].cpu().numpy())
                # 将真实序列转化成文本；reports_ids[:, 1:]去掉了开始符。
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                val_idxs.extend(images_id)
                t.set_description(f'{mode}...')
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            ilog.update(**{f'{mode}_' + k: v for k, v in val_met.items()})
            # 将模型预测结果、真实结果、迭代信息和数据集划分信息保存到一个json文件中
            self._output_generation(val_res, val_gts, val_idxs, epoch, iters, mode)
        return ilog

    def test_step(self, epoch, iters):
        ilog = {'epoch': f'{epoch}-{iters}', 'train_loss': 0.0}

        log = self._test_step(epoch, iters, 'val')
        ilog.update(**(log))

        log = self._test_step(epoch, iters, 'test')
        ilog.update(**(log))

        self._prin_metrics(ilog)
