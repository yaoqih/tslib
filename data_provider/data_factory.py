from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_MarketDaily
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import torch
import random


class MarketDateBatchSampler:
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        dates = dataset.sample_meta["date"].astype(str).to_numpy()
        grouped_indices = {}
        date_order = []
        for idx, date in enumerate(dates):
            if date not in grouped_indices:
                grouped_indices[date] = []
                date_order.append(date)
            grouped_indices[date].append(idx)
        self.batches = [grouped_indices[date] for date in date_order]

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.batches)

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'market_daily': Dataset_MarketDaily,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def market_collate_fn(batch):
    seq_x = torch.stack([item[0] for item in batch], dim=0)
    seq_y = torch.stack([item[1] for item in batch], dim=0)
    seq_x_mark = torch.stack([item[2] for item in batch], dim=0)
    seq_y_mark = torch.stack([item[3] for item in batch], dim=0)
    if len(batch[0]) == 6:
        aux_x = torch.stack([item[4] for item in batch], dim=0)
        meta = torch.tensor([item[5] for item in batch], dtype=torch.long)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, aux_x, meta
    meta = torch.tensor([item[4] for item in batch], dtype=torch.long)
    return seq_x, seq_y, seq_x_mark, seq_y_mark, meta


def _dataloader_kwargs(args):
    num_workers = int(getattr(args, "num_workers", 0))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(getattr(args, "pin_memory", True)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", True))
        kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2))
    return kwargs


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            **_dataloader_kwargs(args),
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            **_dataloader_kwargs(args),
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        collate = market_collate_fn if args.data == 'market_daily' else None
        use_market_cross_section_batches = bool(
            args.data == 'market_daily' and getattr(args, 'market_cross_section_batches', False)
        )
        if use_market_cross_section_batches:
            batch_sampler = MarketDateBatchSampler(data_set, shuffle=shuffle_flag)
            data_loader = DataLoader(
                data_set,
                batch_sampler=batch_sampler,
                **_dataloader_kwargs(args),
                collate_fn=collate,
            )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                **_dataloader_kwargs(args),
                drop_last=drop_last,
                collate_fn=collate)
        return data_set, data_loader
