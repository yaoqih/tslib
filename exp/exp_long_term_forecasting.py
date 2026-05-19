from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.market_multitask import combine_market_multitask_losses, compute_pairwise_rank_loss

warnings.filterwarnings('ignore')


class MarketForecastMultiTaskWrapper(nn.Module):
    def __init__(self, base_model, feature_dim):
        super().__init__()
        self.base_model = base_model
        self.cls_head = nn.Linear(feature_dim, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        forecast = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
        cls_logits = self.cls_head(forecast)
        return {
            'forecast': forecast,
            'cls_logits': cls_logits,
        }


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self._use_market_aux_cls():
            model = MarketForecastMultiTaskWrapper(model, self.args.c_out).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_name = self.args.loss.upper()
        if loss_name == 'MSE':
            return nn.MSELoss()
        if loss_name == 'MAE':
            return nn.L1Loss()
        if loss_name == 'HUBER':
            return nn.HuberLoss(delta=getattr(self.args, 'huber_delta', 1.0))
        raise ValueError(f'Unsupported loss: {self.args.loss}')

    def _use_market_aux_cls(self):
        return bool(
            getattr(self.args, 'market_aux_cls', False)
            and self.args.data == 'market_daily'
            and self.args.task_name == 'long_term_forecast'
        )

    def _use_market_rank_loss(self):
        return bool(
            getattr(self.args, 'market_rank_loss', False)
            and self.args.data == 'market_daily'
            and self.args.task_name == 'long_term_forecast'
        )

    def _split_model_outputs(self, outputs):
        if isinstance(outputs, dict):
            return outputs['forecast'], outputs.get('cls_logits')
        return outputs, None

    def _build_market_cls_targets(self, batch_meta, dataset):
        indices = batch_meta.detach().cpu().numpy().astype(np.int64)
        cls_values = dataset.sample_cls_labels[indices].reshape(-1, self.args.pred_len, 1)
        return torch.tensor(cls_values, dtype=torch.float32, device=self.device)

    def _compute_market_loss(self, outputs, batch_y, batch_meta, dataset, reg_criterion):
        forecast, cls_logits = self._split_model_outputs(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        reg_pred = forecast[:, -self.args.pred_len:, f_dim:]
        reg_true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        reg_loss = reg_criterion(reg_pred, reg_true)
        rank_loss = reg_loss.new_tensor(0.0)
        if self._use_market_rank_loss():
            rank_loss = compute_pairwise_rank_loss(
                pred=reg_pred.reshape(-1),
                target=reg_true.reshape(-1),
                margin=self.args.market_rank_margin,
            )
            reg_loss = reg_loss + self.args.market_rank_weight * rank_loss

        if not self._use_market_aux_cls():
            return {
                'total_loss': reg_loss,
                'reg_loss': reg_loss,
                'cls_loss': reg_loss.new_tensor(0.0),
                'rank_loss': rank_loss,
                'reg_pred': reg_pred,
                'reg_true': reg_true,
            }

        cls_target = self._build_market_cls_targets(batch_meta, dataset)
        cls_loss = self.cls_criterion(
            cls_logits[:, -self.args.pred_len:, :].reshape(-1),
            cls_target.reshape(-1),
        )
        losses = combine_market_multitask_losses(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
            cls_weight=self.args.market_cls_weight,
        )
        return {
            'total_loss': losses['total_loss'],
            'reg_loss': losses['reg_loss'],
            'cls_loss': losses['cls_loss'],
            'rank_loss': rank_loss,
            'reg_pred': reg_pred,
            'reg_true': reg_true,
        }

    def _unpack_batch(self, batch):
        if len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_meta = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_meta
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        return batch_x, batch_y, batch_x_mark, batch_y_mark, None
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_meta = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                losses = self._compute_market_loss(outputs, batch_y, batch_meta, vali_data, criterion)
                loss = losses['total_loss']

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        train_mode = getattr(self.args, 'train_mode', 'best_val')
        use_early_stopping = train_mode == 'best_val'
        if train_mode not in {'best_val', 'fixed_epoch'}:
            raise ValueError(f'Unsupported train_mode: {train_mode}')
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) if use_early_stopping else None

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self._use_market_aux_cls():
            self.cls_criterion = nn.BCEWithLogitsLoss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_meta = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        losses = self._compute_market_loss(outputs, batch_y, batch_meta, train_data, criterion)
                        loss = losses['total_loss']
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    losses = self._compute_market_loss(outputs, batch_y, batch_meta, train_data, criterion)
                    loss = losses['total_loss']
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if use_early_stopping:
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            else:
                torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
                print("Fixed-epoch mode: checkpoint overwritten by current epoch")

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        sample_ids = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        enable_visual = not hasattr(test_data, 'build_prediction_frame')
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_meta = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                forecast, _ = self._split_model_outputs(outputs)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = forecast[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if batch_meta is not None:
                    sample_ids.append(batch_meta.detach().cpu().numpy())
                if enable_visual and i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if hasattr(test_data, 'build_prediction_frame') and sample_ids:
            pred_frame = test_data.build_prediction_frame(
                np.concatenate(sample_ids, axis=0),
                preds[:, :, -1:],
                trues[:, :, -1:],
            )
            market_metrics = test_data.evaluate_predictions(pred_frame)
            pred_frame.to_csv(os.path.join(folder_path, 'top1_predictions.csv'), index=False)
            with open(os.path.join(folder_path, 'market_metrics.txt'), 'w') as mf:
                for key, value in market_metrics.items():
                    mf.write(f'{key}: {value}\n')
            print('market metrics:', market_metrics)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
