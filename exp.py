from yolox.exp import Exp as MyExp
import torch
from yolox.data import COCODataset

class Exp(MyExp):

    def __init__(self):
        self.start_epoch = 0
        self.num_classes = 5
        self.class_names = ["cell", "merged_cell", "row", "column", "table"]
        self.depth = 0.33
        self.width = 0.50

        self.data_dir = "/content/table_dataset"
        self.train_ann = "train_annotations.json"
        self.val_ann = "val_annotations.json"
        self.act = "silu"  # Add this line
        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 5

        self.basic_lr_per_img = 0.01 / 64
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.warmup_epochs = 5
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.min_lr_ratio = 0.05
        self.test_conf = 0.01  # 이 줄을 추가
        self.nmsthre = 0.65
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.flip_prob = 0.5
        self.hsv_prob = 1.0

        self.input_size = (640, 640)
        self.test_size = (640, 640)

        self.print_interval = 5
        self.exp_name = "YOLOX_S_Table_Detection"
        self.save_interval = 5

        self.warmup_lr = 0
        self.ema = True


    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import TrainTransform

        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache_img,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler



        train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
