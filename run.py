import argparse
import os
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='delta for Huber loss')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--train_mode', type=str, default='best_val',
                        help='checkpoint selection mode: best_val, fixed_epoch, or train_loss_plateau')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--train_plateau_patience', type=int, default=3, help='patience for train-loss plateau mode')
    parser.add_argument('--train_plateau_delta', type=float, default=0.0, help='min improvement for train-loss plateau mode')
    parser.add_argument('--train_plateau_ema_decay', type=float, default=0.7, help='EMA decay for train-loss plateau mode')
    parser.add_argument('--train_plateau_metric', type=str, default='loss',
                        help='monitor used by train_loss_plateau: loss, top1_return, topk_mean_return, or rank_ic')
    parser.add_argument('--train_plateau_metric_k', type=int, default=3,
                        help='top-k used when train_plateau_metric=topk_mean_return')
    parser.add_argument('--stage2_epochs', type=int, default=0, help='extra finetune epochs after stage1')
    parser.add_argument('--stage2_train_mode', type=str, default='fixed_epoch',
                        help='checkpoint selection mode for stage2: best_val, fixed_epoch, or train_loss_plateau')
    parser.add_argument('--stage2_train_plateau_metric', type=str, default='topk_mean_return',
                        help='monitor used by stage2 train_loss_plateau when enabled')

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # market research
    parser.add_argument('--market_start_year', type=int, default=2010, help='start year for market data')
    parser.add_argument('--market_fold_year', type=int, default=2019, help='test year for rolling market fold')
    parser.add_argument('--market_test_end', type=str, default='', help='override test end date for market fold')
    parser.add_argument('--market_feature_set', type=str, default='A', help='market feature set id')
    parser.add_argument('--market_aux_feature_set', type=str, default='B_MKT',
                        help='market feature set id for auxiliary head')
    parser.add_argument('--market_target_mode', type=str, default='raw',
                        help='market regression target mode: raw, cross_section_rank, or cross_section_rank_weighted')
    parser.add_argument('--market_train_horizons', type=str, default='1,3,5',
                        help='comma-separated market training horizons: 1 uses tradable open-open label, 3/5 use close-close labels')
    parser.add_argument('--market_train_horizon_weights', type=str, default='2.0,0.25,0.25',
                        help='comma-separated weights for market_train_horizons')
    parser.add_argument('--market_min_history', type=int, default=120, help='min listing history in trading days')
    parser.add_argument('--market_min_avg_amount', type=float, default=2e7, help='min 20d avg amount')
    parser.add_argument('--market_cache_path', type=str, default='./cache/market_daily_features.parquet',
                        help='cached feature parquet path')
    parser.add_argument('--market_score_debias', type=str, default='none',
                        help='prediction score debias method at evaluation time: none or expanding_mean')
    parser.add_argument('--market_score_debias_strength', type=float, default=1.0,
                        help='strength for evaluation-time score debias')
    parser.add_argument('--market_eval_topk_list', type=str, default='1,3,5',
                        help='comma-separated top-k basket sizes for market evaluation')
    parser.add_argument('--market_cross_section_batches', action='store_true', default=False,
                        help='group market_daily batches by date for cross-sectional training')
    parser.add_argument('--market_train_full_window', action='store_true', default=False,
                        help='expand market train split to include the original validation year')
    parser.add_argument('--market_aux_cls', action='store_true', default=False,
                        help='enable auxiliary BCE classification head for market forecasting')
    parser.add_argument('--market_cls_weight', type=float, default=0.5,
                        help='loss weight for auxiliary market BCE head')
    parser.add_argument('--market_rank_loss', action='store_true', default=False,
                        help='enable pairwise rank loss for market forecasting')
    parser.add_argument('--market_rank_weight', type=float, default=0.1,
                        help='loss weight for pairwise rank loss')
    parser.add_argument('--market_rank_margin', type=float, default=0.0,
                        help='margin for pairwise rank loss')
    parser.add_argument('--market_rank_min_target_gap', type=float, default=0.0,
                        help='minimum target gap for same-day pairwise rank loss')
    parser.add_argument('--market_topk_loss', action='store_true', default=False,
                        help='enable top-focused listwise loss for same-day market ranking')
    parser.add_argument('--market_topk_weight', type=float, default=0.0,
                        help='loss weight for top-focused listwise loss')
    parser.add_argument('--market_topk_k', type=int, default=3,
                        help='top-k size for top-focused listwise loss')
    parser.add_argument('--market_topk_temperature', type=float, default=1.0,
                        help='temperature for top-focused listwise loss')
    parser.add_argument('--market_topk_target_mode', type=str, default='soft',
                        help='target mode for top-focused listwise loss: soft or hard')
    parser.add_argument('--market_regression_use_tradable_mask', action='store_true', default=False,
                        help='apply tradable mask to market regression loss in addition to rank/top-k losses')
    parser.add_argument('--market_pred_topq_ratio', type=float, default=0.0,
                        help='upweight the current predicted top-q fraction of stocks during training')
    parser.add_argument('--market_pred_topq_weight', type=float, default=1.0,
                        help='sample weight assigned to the current predicted top-q fraction')
    parser.add_argument('--market_head_candidate_ratio', type=float, default=0.0,
                        help='upweight the union of predicted and true top-q head candidates during training')
    parser.add_argument('--market_head_candidate_weight', type=float, default=1.0,
                        help='sample weight assigned to the union head-candidate set')
    parser.add_argument('--market_winner_loss', action='store_true', default=False,
                        help='enable head-union winner-selection pairwise loss')
    parser.add_argument('--market_winner_weight', type=float, default=0.0,
                        help='loss weight for winner-selection pairwise loss')
    parser.add_argument('--market_winner_topq_ratio', type=float, default=0.2,
                        help='top-q ratio used to build the predicted-vs-true head union for winner loss')
    parser.add_argument('--market_winner_margin', type=float, default=0.0,
                        help='margin for winner-selection pairwise loss')
    parser.add_argument('--market_winner_min_target_gap', type=float, default=0.0,
                        help='minimum target gap used by winner-selection pairwise loss')
    parser.add_argument('--market_local_loss', action='store_true', default=False,
                        help='enable local comparable-set pairwise rank loss')
    parser.add_argument('--market_local_weight', type=float, default=0.0,
                        help='loss weight for local comparable-set pairwise rank loss')
    parser.add_argument('--market_local_neighbor_k', type=int, default=20,
                        help='number of dynamic nearest neighbors used by the local comparable-set loss')
    parser.add_argument('--market_local_margin', type=float, default=0.0,
                        help='margin for local comparable-set pairwise rank loss')
    parser.add_argument('--market_local_min_target_gap', type=float, default=0.0,
                        help='minimum target gap used by local comparable-set pairwise rank loss')
    parser.add_argument('--market_local_feature_names', type=str, default='log_amount,turnover_rate,amplitude,ret_20,vol_20',
                        help='comma-separated market feature names used to build the local comparable-set graph')
    parser.add_argument('--market_true_rank_alpha', type=float, default=3.0,
                        help='extra weight scale for true cross-sectional head names in weighted rank mode')
    parser.add_argument('--market_true_rank_power', type=float, default=2.0,
                        help='power used for true-rank sample weighting in weighted rank mode')
    parser.add_argument('--market_aux_rank_reg_weight', type=float, default=0.1,
                        help='small auxiliary regression weight against rank target in weighted rank mode')
    parser.add_argument('--market_mixed_rank_weight', type=float, default=1.0,
                        help='rank-loss weight for cs_two_stage candidate head')
    parser.add_argument('--market_mixed_reg_weight', type=float, default=1.0,
                        help='winner regression weight for cs_two_stage winner head')
    parser.add_argument('--market_winner_topk', type=int, default=20,
                        help='candidate-pool size used by cs_two_stage winner head')
    parser.add_argument('--market_train_on_tradable_only', action='store_true', default=False,
                        help='apply tradable mask to all market training losses when enabled')
    parser.add_argument('--market_head_concentration_weight', type=float, default=0.0,
                        help='loss weight for penalizing overly concentrated same-day score mass')
    parser.add_argument('--market_head_concentration_temperature', type=float, default=1.0,
                        help='temperature for same-day score softmax used by head concentration penalty')
    parser.add_argument('--market_head_gap_weight', type=float, default=0.0,
                        help='loss weight for penalizing overly large top1-vs-head score gap')
    parser.add_argument('--market_head_gap_topk', type=int, default=3,
                        help='head basket size used by the top-gap penalty')
    parser.add_argument('--market_static_bias_weight', type=float, default=0.0,
                        help='loss weight for no-state static-bias surrogate penalty')
    parser.add_argument('--market_static_bias_topk', type=int, default=3,
                        help='head basket size used by the no-state static-bias surrogate penalty')
    parser.add_argument('--stage2_rank_weight', type=float, default=0.1,
                        help='stage2 loss weight for pairwise rank loss')
    parser.add_argument('--stage2_topk_weight', type=float, default=0.0,
                        help='stage2 loss weight for top-focused listwise loss')
    parser.add_argument('--stage2_topk_k', type=int, default=3,
                        help='stage2 top-k size for top-focused listwise loss')
    parser.add_argument('--stage2_topk_temperature', type=float, default=1.0,
                        help='stage2 temperature for top-focused listwise loss')
    parser.add_argument('--stage2_topk_target_mode', type=str, default='soft',
                        help='stage2 target mode for top-focused listwise loss: soft or hard')
    parser.add_argument('--stage2_head_concentration_weight', type=float, default=0.0,
                        help='stage2 loss weight for same-day score concentration penalty')
    parser.add_argument('--stage2_head_gap_weight', type=float, default=0.0,
                        help='stage2 loss weight for same-day top1-vs-head gap penalty')
    parser.add_argument('--stage2_head_concentration_temperature', type=float, default=1.0,
                        help='stage2 temperature for same-day score concentration penalty')
    parser.add_argument('--stage2_static_bias_weight', type=float, default=0.0,
                        help='stage2 loss weight for no-state static-bias surrogate penalty')
    parser.add_argument('--stage2_static_bias_topk', type=int, default=3,
                        help='stage2 head basket size for no-state static-bias surrogate penalty')
    parser.add_argument('--market_cs_layers', type=int, default=1,
                        help='number of stock-axis self-attention layers for market cross-section model')
    parser.add_argument('--market_cs_n_heads', type=int, default=4,
                        help='number of stock-axis attention heads for market cross-section model')
    parser.add_argument('--market_cs_d_ff', type=int, default=256,
                        help='ffn width inside the market cross-section attention block')
    parser.add_argument('--market_cs_dropout', type=float, default=0.1,
                        help='dropout for the market cross-section attention block')
    parser.add_argument('--market_cs_recent_k', type=int, default=5,
                        help='number of recent latent tokens kept by the cross-section head')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)


    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            
            # Override setting for specific model to ensure proper checkpoint naming and logging
            if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
                setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}'

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        
        # Override setting for specific model to ensure proper checkpoint naming and logging
        if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
            setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{ii}'

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
