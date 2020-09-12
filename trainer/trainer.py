import time

import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from utils.util import AverageMeter


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
    ):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar("{}".format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        end_time = time.time()
        for batch_idx, (input_ids, attention_mask, segment_ids, target) in enumerate(
            self.data_loader
        ):
            data_time.update(time.time() - end_time)

            target = target.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            output = self.model(batch=(input_ids, attention_mask, segment_ids))
            loss = self.loss(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar("loss", loss.item())
            self.writer.add_scalar("loss", loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            losses.update(loss.item(), target.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Epoch: [{0}][{1}/{2}] {percent:.0f}%\t lr: {lr:.5f}\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                        epoch,
                        batch_idx,
                        len(self.data_loader),
                        percent=100.0 * batch_idx / len(self.data_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                    )
                )

        log = {
            "loss": total_loss / len(self.data_loader),
            "metrics": (total_metrics / len(self.data_loader)).tolist(),
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for (
                batch_idx,
                (input_ids, attention_mask, segment_ids, target),
            ) in enumerate(self.valid_data_loader):
                target = target.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)

                output = self.model(batch=(input_ids, attention_mask, segment_ids))
                loss = self.loss(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.writer.add_scalar("loss", loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return {
            "val_loss": total_val_loss / len(self.valid_data_loader),
            "val_metrics": (total_val_metrics / len(self.valid_data_loader)).tolist(),
        }