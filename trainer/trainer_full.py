import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer


class DefaultTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler, resume, data_loader,
                 valid_data_loader=None, train_logger=None, **extra_args):
        super(DefaultTrainer, self).__init__(config, model, loss, metrics, optimizer, lr_scheduler, resume,
                                             train_logger)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.img_size = data_loader.img_size
        self.num_points = data_loader.num_points
        self.G = (self.img_size[0] * self.img_size[1]) // self.num_points

    def _eval_metrics(self, S0_pred, S0, k_pred, k, theta_pred, theta):
        current_metrics = dict()
        for variable_name, mets in self.metrics.items():
            # manually mapping
            if variable_name == 'S0':
                x, y = S0_pred, S0
            elif variable_name == 'p':
                x, y = k_pred.abs(), k.abs()
            elif variable_name == 'theta':
                x, y = theta_pred, theta
            else:
                raise Exception(f'variable_name: {variable_name} not found in metrics!')

            # calculate and record current_metrics
            current_metrics[variable_name] = np.zeros(len(mets))
            for i, met in enumerate(mets):
                m = met(x, y)
                current_metrics[variable_name][i] += m
                self.writer.add_scalar(f'{met.__name__}_{variable_name}', m)

        return current_metrics

    @staticmethod
    def _update_total_metrics(total_metrics, current_metrics):
        for variable_name in total_metrics:
            total_metrics[variable_name] += current_metrics[variable_name]

    @staticmethod
    def _avg_metrics(total_metrics, l):
        return {key: value / l for key, value in total_metrics.items()}

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
        # set the model to train mode
        self.model.train()

        total_loss = 0
        total_metrics = dict()
        for variable_name, mets in self.metrics.items():
            total_metrics[variable_name] = np.zeros(len(mets))

        # start training
        for batch_idx, sample in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            assert batch_idx == 0

            # get data and normalize
            S0 = sample['S0'].to(self.device) / 2  # (1 3 H W), real
            k = sample['k'].to(self.device)  # (1 3 H W), complex
            theta = sample['theta'].to(self.device) / np.pi  # (1 3 H W), real
            flattened_xy = sample['flattened_xy'].to(self.device)  # (1 G num_points 2), real
            flattened_S0 = sample['flattened_S0'].to(self.device) / 2  # (1 G num_points 3), real
            flattened_k = sample['flattened_k'].to(self.device)  # (1 G num_points 3), complex
            flattened_theta = sample['flattened_theta'].to(self.device) / np.pi  # (1 G num_points 3), real

            # temporarily store the flattened predictions for rendering the full results
            flattened_S0_pred = torch.zeros_like(flattened_S0)  # (1 G num_points 3), real
            flattened_k_pred = torch.zeros_like(flattened_k)  # (1 G num_points 3), complex
            flattened_theta_pred = torch.zeros_like(flattened_theta)  # (1 G num_points 3), real

            # used for storing the loss
            model_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

            # iterate each group of points
            for i in range(0, self.G):
                xy_group = flattened_xy[:, i:i + 1, :, :]  # (1 1 num_points 2), real
                S0_group = flattened_S0[:, i:i + 1, :, :]  # (1 1 num_points 3), real
                k_group = flattened_k[:, i:i + 1, :, :]  # (1 1 num_points 3), complex
                theta_group = flattened_theta[:, i:i + 1, :, :]  # (1 1 num_points 3), real

                # infer
                S0_group_pred, k_group_pred, theta_group_pred = self.model(xy_group)  # (1 1 num_points 3)

                # calculate group_loss
                group_loss = self.loss(S0_group_pred, S0_group, k_group_pred, k_group, theta_group_pred, theta_group)

                # train model
                self.optimizer.zero_grad()
                group_loss.backward()
                self.optimizer.step()

                # store the results
                flattened_S0_pred[:, i:i + 1, :, :] = S0_group_pred
                flattened_k_pred[:, i:i + 1, :, :] = k_group_pred
                flattened_theta_pred[:, i:i + 1, :, :] = theta_group_pred
                # update model_loss
                model_loss += group_loss

            # unflatten the flattened predictions
            S0_pred = flattened_S0_pred.permute(0, 3, 1, 2).reshape(1, 3, self.img_size[0],
                                                                    self.img_size[1])  # (1 3 H W), real
            k_pred = flattened_k_pred.permute(0, 3, 1, 2).reshape(1, 3, self.img_size[0],
                                                                  self.img_size[1])  # (1 3 H W), complex
            theta_pred = flattened_theta_pred.permute(0, 3, 1, 2).reshape(1, 3, self.img_size[0],
                                                                          self.img_size[1])  # (1 3 H W), real

            # calculate total loss/metrics and add scalar to tensorboard
            model_loss /= self.G
            self.writer.add_scalar('loss', model_loss.item())
            total_loss += model_loss.item()
            current_metrics = self._eval_metrics(S0_pred, S0, k_pred, k, theta_pred, theta)
            self._update_total_metrics(total_metrics, current_metrics)

            # show current training info
            if self.verbosity >= 2:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        model_loss,
                    )
                )

        # turn the learning rate
        self.lr_scheduler.step()

        # get batch average loss/metrics as log and do validation
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': self._avg_metrics(total_metrics, len(self.data_loader))
        }

        if self.do_validation:
            # let's do validation in every 200 epochs
            if epoch % 200 == 0:
                val_log = self._valid_epoch(epoch)
                log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training several epochs

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        # set the model to validation mode
        self.model.eval()

        total_loss = 0
        total_metrics = dict()
        for variable_name, mets in self.metrics.items():
            total_metrics[variable_name] = np.zeros(len(mets))

        # start validating
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

                # get data
                S0 = sample['S0'].to(self.device) / 2  # (1 3 H W), real
                p = sample['p'].to(self.device)  # (1 3 H W), real
                theta = sample['theta'].to(self.device) / np.pi  # (1 3 H W), real
                k = sample['k'].to(self.device)  # (1 3 H W), complex
                flattened_xy = sample['flattened_xy'].to(self.device)  # (1 G num_points 2), real
                flattened_S0 = sample['flattened_S0'].to(self.device) / 2  # (1 G num_points 3), real
                flattened_k = sample['flattened_k'].to(self.device)  # (1 G num_points 3), complex
                flattened_theta = sample['flattened_theta'].to(self.device) / np.pi  # (1 G num_points 3), real

                # temporarily store the flattened predictions for rendering the full results
                flattened_S0_pred = torch.zeros_like(flattened_S0)  # (1 G num_points 3), real
                flattened_k_pred = torch.zeros_like(flattened_k)  # (1 G num_points 3), complex
                flattened_theta_pred = torch.zeros_like(flattened_theta)  # (1 G num_points 3), real

                # used for storing the loss
                model_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

                # iterate each group of points
                for i in range(0, self.G):
                    xy_group = flattened_xy[:, i:i + 1, :, :]  # (1 1 num_points 2), real
                    S0_group = flattened_S0[:, i:i + 1, :, :]  # (1 1 num_points 3), real
                    k_group = flattened_k[:, i:i + 1, :, :]  # (1 1 num_points 3), complex
                    theta_group = flattened_theta[:, i:i + 1, :, :]  # (1 1 num_points 3), real

                    # infer
                    S0_group_pred, k_group_pred, theta_group_pred = self.model(xy_group)  # (1 1 num_points 3)

                    # calculate group_loss
                    group_loss = self.loss(S0_group_pred, S0_group, k_group_pred, k_group, theta_group_pred,
                                           theta_group)

                    # store the results
                    flattened_S0_pred[:, i:i + 1, :, :] = S0_group_pred
                    flattened_k_pred[:, i:i + 1, :, :] = k_group_pred
                    flattened_theta_pred[:, i:i + 1, :, :] = theta_group_pred
                    # update model_loss
                    model_loss += group_loss

                # unflatten the flattened predictions
                S0_pred = flattened_S0_pred.permute(0, 3, 1, 2).reshape(1, 3, self.img_size[0],
                                                                        self.img_size[1])  # (1 3 H W), real
                k_pred = flattened_k_pred.permute(0, 3, 1, 2).reshape(1, 3, self.img_size[0],
                                                                      self.img_size[1])  # (1 3 H W), complex
                theta_pred = flattened_theta_pred.permute(0, 3, 1, 2).reshape(1, 3, self.img_size[0],
                                                                              self.img_size[1])  # (1 3 H W), real

                # calculate total loss/metrics and add scalar to tensorboard
                model_loss /= self.G
                self.writer.add_scalar('loss', model_loss.item())
                total_loss += model_loss.item()
                current_metrics = self._eval_metrics(S0_pred, S0, k_pred, k, theta_pred, theta)
                self._update_total_metrics(total_metrics, current_metrics)

                # visualization
                # during training we simply visualize p and theta without color map
                self.writer.add_image('S0', make_grid(S0))
                self.writer.add_image('S0_pred', make_grid(S0_pred))
                self.writer.add_image('p', make_grid(p))
                self.writer.add_image('p_pred', make_grid(k_pred.abs()))
                self.writer.add_image('theta', make_grid(theta))
                self.writer.add_image('theta_pred', make_grid(theta_pred))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_loss / len(self.valid_data_loader),
            'val_metrics': self._avg_metrics(total_metrics, len(self.valid_data_loader))
        }
