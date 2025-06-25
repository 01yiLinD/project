import os
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from datasets import SpineCTDataset
from models import build_model
from torch.optim import AdamW
from torch.amp import autocast
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
    handlers=[
        logging.FileHandler("training.log"),  # 将日志输出到文件
        logging.StreamHandler()  # 将日志输出到控制台
    ]
)
logger = logging.getLogger(__name__)  # 创建日志记录器

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message)

    def flush(self):
        pass

# 超参数设置
class Args:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_queries = 25
        self.aux_loss = True  # 确保辅助损失被启用
        self.dec_layers = 2
        self.hidden_dim = 384
        self.nheads = 8
        self.dim_feedforward = 2048
        self.dropout = 0.8
        self.enc_layers = 2
        self.pre_norm = False
        self.ce_loss_coef = 1.0
        self.bbox_loss_coef = 2.0
        self.giou_loss_coef = 1.0
        self.edges_loss_coef = 2.0  # 新增edges损失系数
        self.feature_map_dim = (12, 12, 12)
        self.patch_size = 192
        self.lr_backbone = 1e-5
        self.lr = 1e-5
        self.weight_decay = 1e-4
        self.epochs = 200
        self.batch_size = 1
        self.voxel_spacing = (1, 1, 1)

        # # autodl
        # self.train_rawdata_dir = "/root/autodl-tmp/data/train/rawdata"
        # self.train_derivatives_dir = "/root/autodl-tmp/data/train/derivatives"
        # self.val_rawdata_dir = "/root/autodl-tmp/data/validation/rawdata"
        # self.val_derivatives_dir = "/root/autodl-tmp/data/validation/derivatives"
        # self.test_rawdata_dir = "/root/autodl-tmp/data/test/rawdata"
        # self.test_derivatives_dir = "/root/autodl-tmp/data/test/derivatives"

        # # local
        # self.train_rawdata_dir = "./data/train/rawdata"
        # self.train_derivatives_dir = "./data/train/derivatives"
        # self.val_rawdata_dir = "./data/validation/rawdata"
        # self.val_derivatives_dir = "./data/validation/derivatives"
        # self.test_rawdata_dir = "./data/test/rawdata"
        # self.test_derivatives_dir = "./data/test/derivatives"

        # huangr
        self.train_rawdata_dir = "./data/huangr/VerSe/train/rawdata/"
        self.train_derivatives_dir = "./data/huangr/VerSe/train/derivatives/"
        self.val_rawdata_dir = "./data/huangr/VerSe/validation/rawdata/"
        self.val_derivatives_dir = "./data/huangr/VerSe/validation/derivatives/"
        self.test_rawdata_dir = "./data/test/rawdata"
        self.test_derivatives_dir = "./data/test/derivatives"

        self.resample_spacing = 1.0


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.criterion, _ = build_model(cfg)
        self.model = self.model.to(cfg.device)

        # 优化器配置
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": cfg.lr_backbone},
        ]
        self.optimizer = AdamW(
            param_dicts,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        # 数据集初始化
        train_dataset = SpineCTDataset(
            rawdata_dir=cfg.train_rawdata_dir,  # 修正路径参数
            derivatives_dir=cfg.train_derivatives_dir,
            patch_size=cfg.patch_size,
            resample_spacing=cfg.resample_spacing
        )
        val_dataset = SpineCTDataset(  # 验证数据集
            rawdata_dir=cfg.val_rawdata_dir,
            derivatives_dir=cfg.val_derivatives_dir,
            patch_size=cfg.patch_size,
            resample_spacing=cfg.resample_spacing
        )

        self.train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                      num_workers=4, pin_memory=True, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                    num_workers=4, pin_memory=True)

        self.loss_history = {"train": [], "val": []}
        self.best_loss = float('inf')
        self.checkpoint_dir = "./checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_path):
        """
        加载模型的检查点
        :param checkpoint_path: 检查点文件的路径
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.cfg.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1  # 从下一个 epoch 开始
        self.best_loss = checkpoint.get("val_loss", float('inf'))
        self.loss_history["train"] = checkpoint.get("train_loss_history", [])
        self.loss_history["val"] = checkpoint.get("val_loss_history", [])
        logger.info(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {start_epoch}.")
        return start_epoch

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        scaler = torch.amp.GradScaler(enabled=True)  # 修正GradScaler初始化

        for batch_idx, (samples, masks, targets) in enumerate(self.train_loader):
            samples = samples.to(self.cfg.device)
            masks = masks.to(self.cfg.device)
            targets = {k: v.to(self.cfg.device) for k, v in targets.items()}

            self.optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):  # 修正autocast使用
                outputs = self.model(samples)
                # 计算所有输出层的损失（包括辅助损失）
                loss_dict = self.criterion(outputs[0], targets)  # 传递整个outputs

                losses = (
                    loss_dict['loss_ce'] * self.cfg.ce_loss_coef +
                    loss_dict['loss_bbox'] * self.cfg.bbox_loss_coef +
                    loss_dict['loss_giou'] * self.cfg.giou_loss_coef +
                    loss_dict.get('loss_edges_sphere', 0) * self.cfg.edges_loss_coef  # 处理可选损失项
                )

            scaler.scale(losses).backward()
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += losses.item()
            del samples, targets, outputs
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(self.train_loader)
        self.loss_history["train"].append(avg_loss)
        logger.info(f"Training Loss: {avg_loss:.4f}")

        self.scheduler.step()

    def validate(self):
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (samples, masks, targets) in enumerate(self.val_loader):
                samples = samples.to(self.cfg.device)
                masks = masks.to(self.cfg.device)
                targets = {k: v.to(self.cfg.device) for k, v in targets.items()}

                # 使用 autocast 确保混合精度计算的兼容性
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(samples)
                    # 计算所有输出层的损失（包括辅助损失）
                    loss_dict = self.criterion(outputs[0], targets)  # 传递整个 outputs

                losses = (
                    loss_dict['loss_ce'] * self.cfg.ce_loss_coef +
                    loss_dict['loss_bbox'] * self.cfg.bbox_loss_coef +
                    loss_dict['loss_giou'] * self.cfg.giou_loss_coef +
                    loss_dict.get('loss_edges_sphere', 0) * self.cfg.edges_loss_coef
                )
                total_val_loss += losses.item()

                del samples, targets, outputs
                torch.cuda.empty_cache()

        avg_val_loss = total_val_loss / len(self.val_loader)
        self.loss_history["val"].append(avg_val_loss)
        logger.info(f"Validation Loss: {avg_val_loss: .4f}")
        return avg_val_loss

    def save_checkpoint(self, epoch, val_loss):
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"model_epoch_{epoch}_valloss_{val_loss:.2f}.pth"
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss_history": self.loss_history["train"],
            "val_loss_history": self.loss_history["val"],
            "val_loss": val_loss
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def train(self, resume_from_checkpoint=None):
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint(resume_from_checkpoint)

        for epoch in range(start_epoch, self.cfg.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.cfg.epochs}")
            self.train_one_epoch()
            val_loss = self.validate()

            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

        logger.info("Training completed.")


class Tester:
    def __init__(self, cfg, checkpoint_path):
        self.cfg = cfg
        self.device = cfg.device
        self.model, _, _ = build_model(cfg)
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 加载测试数据
        test_dataset = SpineCTDataset(
            rawdata_dir=cfg.test_rawdata_dir,
            derivatives_dir=cfg.test_derivatives_dir,
            patch_size=cfg.patch_size,
            resample_spacing=cfg.resample_spacing
        )
        self.test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=4)

    def test(self):
        import matplotlib.pyplot as plt
        import numpy as np

        all_predictions = []
        result_dir = "result_test"
        os.makedirs(result_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, (samples, masks, targets) in enumerate(self.test_loader):
                samples = samples.to(self.device)
                masks = masks.to(self.device)
                
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(samples)

                # 获取分类预测结果
                pred_logits = outputs[0]['pred_logits']  # 预测的分类分数
                pred_labels = torch.argmax(pred_logits, dim=-1)  # 取最大值索引作为类别
                mask_for_key_points = (pred_labels == 0).unsqueeze(-1)  # shape: [1, 25, 1]
                pred_key_points = outputs[0]['pred_boxes']
                pred_key_points = torch.where(mask_for_key_points, torch.zeros_like(pred_key_points), pred_key_points) * 192
                print(f'key: {pred_key_points}')

                all_predictions.extend(pred_labels.cpu().numpy())
                
                # 可视化：仅对第一个 batch 的第一个样本展示
                if batch_idx == 0:
                    ct_data = samples[0, 0].cpu().numpy()  # [H, W, D]
                    slice_idx = ct_data.shape[2] // 2  # 选取 z 轴中间切片
                    im_np_sag = ct_data[:, :, slice_idx]  # 矢状面切片
                    print(f'im_np_sag.shape: {im_np_sag.shape}')

                    # 获取关键点
                    valid_keypoints = pred_key_points[0].cpu().numpy()  # shape: [25, 4]
                    valid_labels = pred_labels[0].cpu().numpy()  # shape: [25]

                    keypoints_x = valid_keypoints[:, 1]
                    keypoints_y = valid_keypoints[:, 0]

                    # 只保留类别不为 0 的点
                    mask = valid_labels != 0
                    keypoints_x = keypoints_x[mask]
                    keypoints_y = keypoints_y[mask]

                    # 创建一个新的图像来绘制原图和关键点
                    im_np_sag = (im_np_sag - np.min(im_np_sag)) / (np.max(im_np_sag) - np.min(im_np_sag))
                    plt.imshow(im_np_sag, cmap="gray")

                    # 在原图上叠加关键点
                    plt.scatter(keypoints_x, keypoints_y, c="red", marker="o", s=30, label="Keypoints", edgecolors='black', linewidth=1)

                    # 设置标题，添加图例
                    plt.title("CT Image with Keypoints")
                    plt.legend()
                    plt.axis("off")

                    # 保存可视化结果
                    output_path = os.path.join(result_dir, "keypoints_sagittal.png")
                    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
                    plt.close()

                    logger.info(f"Keypoints visualization saved to {output_path}")

                # 释放显存
                del samples, masks, targets, outputs
                torch.cuda.empty_cache()

        # 将预测结果写入 result.txt
        result_text_path = os.path.join(result_dir, "result.txt")
        with open(result_text_path, "w") as f:
            for pred in all_predictions:
                f.write(str(pred) + "\n")

        logger.info(f"Test Predictions: {all_predictions}")
        return all_predictions



if __name__ == "__main__":
    # 重定向标准输出和错误
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)
    
    cfg = Args()
    # trainer = Trainer(cfg)
    
    checkpoint_path = "./checkpoints/model_epoch_97_valloss_1.63.pth"

    # if os.path.exists(checkpoint_path):
        #trainer.train(resume_from_checkpoint=checkpoint_path)
    # else:
        #trainer.train()
        
    tester = Tester(cfg, checkpoint_path)
    tester.test()