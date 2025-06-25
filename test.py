from scipy import ndimage
import torch
from models import build_model
import nibabel as nib
from util.data_utilities import reorient_to, resample_nib
from train import Args
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import logging
from torch.amp import autocast
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    

class SpineCTDataset(Dataset):
    def __init__(self,
                 test_dir: str,
                 patch_size: int = 192,
                 resample_spacing: float = 1.0,
                 vertebra_range: tuple = (1, 25),
                 max_queries: int = 25):

        self.test_dir = Path(test_dir)
        self.patch_size = patch_size
        self.resample_spacing = resample_spacing
        self.vertebra_range = vertebra_range
        self.max_queries = max_queries
        self.orientation = ('I', 'P', 'L')  # 标准方向
        # 加载有效病例数据
        self.case_data = self._validate_and_load_cases()
        logger.info(f"成功加载 {len(self.case_data)} 个有效病例")

    def _validate_and_load_cases(self):
        """验证并加载有效病例数据"""
        valid_cases = []

        # 遍历原始数据目录
        for root, _, files in os.walk(self.test_dir):
            for file in files:
                if file.endswith(".nii.gz"):
                    # 解析病例信息
                    case_id = file.split("_")[0]
                    ct_path = Path(root) / file

                    valid_cases.append({
                        "case_id": case_id,
                        "ct_path": ct_path,
                    })

        if not valid_cases:
            raise RuntimeError("未找到任何有效病例数据！")

        return valid_cases

    def __len__(self):
        return len(self.case_data)  # 每个病例只采样1个块（整个图像）

    def __getitem__(self, idx):
        case = self.case_data[idx]

        try:

            img_nib = nib.load(case["ct_path"])
            img_iso = resample_nib(img_nib, voxel_spacing=(self.resample_spacing,)*3, order=3)
            img_iso = reorient_to(img_iso, axcodes_to=self.orientation)

            # => 192 x 192 x 192
            orig_shape = img_iso.shape

            ct_data = ndimage.zoom(img_iso.get_fdata(),
                                   [self.patch_size/orig_shape[0],
                                    self.patch_size/orig_shape[1],
                                    self.patch_size/orig_shape[2]],
                                   order=1,
                                   mode='constant',
                                   cval=-1024)


            ct_data = np.clip(ct_data, -1000, 1500)
            ct_data = (ct_data + 1000) / 2500


            patch_tensor = torch.from_numpy(ct_data).float().unsqueeze(0)  # [1, H, W, D]

            return patch_tensor

        except Exception as e:
            logger.error(f"处理病例 {case['case_id']} 时出错: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))  # 跳过错误样本
    


class Tester:
    def __init__(self, cfg, checkpoint_path):
        self.cfg = cfg
        self.device = cfg.device
        self.model, _, _ = build_model(cfg)
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 加载测试数据
        test_dataset = SpineCTDataset(
            test_dir=Path("data/train/derivatives"),
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
            for batch_idx, samples in enumerate(self.test_loader):
                samples = samples.to(self.device)
                
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
                
                # # 可视化：仅对第一个 batch 的第一个样本展示
                if batch_idx == 0:
                    ct_data = samples[0, 0].cpu().numpy()  # [H, W, D]
                    slice_idx = ct_data.shape[2] // 2  # 选取 z 轴中间切片
                    im_np_sag = ct_data[:, :, slice_idx]  # 矢状面切片
                    im_np_sag = (im_np_sag - np.min(im_np_sag)) / (np.max(im_np_sag) - np.min(im_np_sag))
                    plt.imshow(im_np_sag, cmap="gray")
                    # 设置标题，添加图例
                    plt.title("CT Image")
                    plt.legend()
                    plt.axis("off")

                    # 保存可视化结果
                    output_path = os.path.join(result_dir, "keypoints_sagittal.png")
                    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
                    plt.close()

                    logger.info(f"Keypoints visualization with ground truth saved to {output_path}")

                # 释放显存
                del samples, outputs
                torch.cuda.empty_cache()

        logger.info(f"Test Predictions: {all_predictions}")
        return all_predictions
            
    

if __name__ == "__main__":
    cfg = Args()
    checkpoint_path = "checkpoints/model_epoch_97_valloss_1.63.pth"
    tester = Tester(cfg, checkpoint_path)
    tester.test()        
    
