import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from scipy.ndimage import zoom

import sys

current_dir = Path(__file__).parent
project_root = current_dir.parent

sys.path.append(str(project_root))

from util.data_utilities import load_centroids, reorient_to, resample_nib, rescale_centroids, reorient_centroids_to
import logging
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpineCTDataset(Dataset):
    def __init__(self,
                 rawdata_dir: str,
                 derivatives_dir: str,
                 patch_size: int = 192,
                 resample_spacing: float = 1.0,
                 vertebra_range: tuple = (1, 25),
                 max_queries: int = 25):
        """
        BIDS格式脊柱CT数据集加载器

        参数:
        rawdata_dir: 原始数据目录路径
        derivatives_dir: 处理数据目录路径
        patch_size: 采样块尺寸
        resample_spacing: 重采样间距(mm)
        vertebra_range: 关注的椎体标签范围 (start, end)
        max_queries: 最大椎体查询数量
        """
        self.rawdata_dir = Path(rawdata_dir)
        self.derivatives_dir = Path(derivatives_dir)
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
        for root, _, files in os.walk(self.rawdata_dir):
            for file in files:
                if file.endswith(".nii.gz"):
                    # 解析病例信息
                    case_id = file.split("_")[0]
                    # print(case_id)

                    # print(case_id)
                    ct_path = Path(root) / file

                    # 检查衍生数据
                    deriv_dir = self.derivatives_dir / case_id
                    if file.split("_")[1] != "ct.nii.gz":
                        deriv_dir = deriv_dir / (case_id + "_" + file.split("_")[1])
                        mask_path = Path(str(deriv_dir) + f"_seg-vert_msk.nii.gz")
                        ctd_path = Path(str(deriv_dir) + f"_seg-subreg_ctd.json")
                    else:
                        mask_path = deriv_dir / f"{case_id}_seg-vert_msk.nii.gz"
                        ctd_path = deriv_dir / f"{case_id}_seg-subreg_ctd.json"
                    # 验证文件存在性
                    missing_files = []
                    if not mask_path.exists():
                        missing_files.append("分割掩码")
                    if not ctd_path.exists():
                        missing_files.append("中心点标注")

                    if missing_files:
                        logger.warning(f"病例 {case_id} 缺少 {'、'.join(missing_files)}，已跳过")
                        continue

                    valid_cases.append({
                        "case_id": case_id,
                        "ct_path": ct_path,
                        "mask_path": mask_path,
                        "ctd_path": ctd_path
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
            msk_nib = nib.load(case["mask_path"])
            ctd_list = load_centroids(case["ctd_path"])

            # Resample & Reorient data
            img_iso = resample_nib(img_nib, voxel_spacing=(self.resample_spacing,)*3, order=3)
            msk_iso = resample_nib(msk_nib, voxel_spacing=(self.resample_spacing,)*3, order=0)
            ctd_iso = rescale_centroids(ctd_list, img_nib, (1, 1, 1))
            # print(f'ctd_iso: {ctd_iso}')

            img_iso = reorient_to(img_iso, axcodes_to=self.orientation)
            msk_iso = reorient_to(msk_iso, axcodes_to=self.orientation)
            ctd_iso = reorient_centroids_to(ctd_iso, img_iso)

            # print(f'ctd_iso: {ctd_iso}')

            # => 192 x 192 x 192
            orig_shape = img_iso.shape

            ct_data = ndimage.zoom(img_iso.get_fdata(),
                                   [self.patch_size/orig_shape[0],
                                    self.patch_size/orig_shape[1],
                                    self.patch_size/orig_shape[2]],
                                   order=1,
                                   mode='constant',
                                   cval=-1024)
            
            mask_data = ndimage.zoom(msk_iso.get_fdata(),
                                     [self.patch_size/orig_shape[0],
                                      self.patch_size/orig_shape[1],
                                      self.patch_size/orig_shape[2]],
                                      order=0,
                                      mode='constant',
                                      cval=0)
            
            scale_factors = np.array([
                self.patch_size/orig_shape[0],
                self.patch_size/orig_shape[1],
                self.patch_size/orig_shape[2]
            ])

            # print(f"scale_factors: {scale_factors}")

            centers = []
            for c in ctd_iso[1:]:

                try:
                    
                    phys_coord = np.array([c[1], c[2], c[3]])
                    scaled_coord = phys_coord * scale_factors

                    x, y, z = np.round(scaled_coord).astype(int)

                    # np.clip：将数值限制在指定范围（0~191）内
                    x = np.clip(scaled_coord[0], 0, self.patch_size-1)
                    y = np.clip(scaled_coord[1], 0, self.patch_size-1)
                    z = np.clip(scaled_coord[2], 0, self.patch_size-1)

                    x, y, z = int(np.round(x)), int(np.round(y)), int(np.round(z))
                    # print(f"x: {x}, y: {y}, z: {z}")

                    # test
                    # print(mask_data[x, y, z])

                    if (self.vertebra_range[0] <= c[0] <= self.vertebra_range[1] and mask_data[x, y, z] == c[0]):
                        centers.append((c[0], x, y, z))
                    
                    # print(f"centers: {centers}")

                except Exception as e:
                    logger.warning(f"坐标转换异常: {c} - {str(e)}")

            ct_data = np.clip(ct_data, -1000, 1500)
            ct_data = (ct_data + 1000) / 2500

            # generate labels
            labels = self._generate_labels(
                patch_shape=(self.patch_size,)*3,
                centers=centers,
                mask_patch=mask_data
            )

            patch_tensor = torch.from_numpy(ct_data).float().unsqueeze(0)  # [1, H, W, D]
            mask_tensor = torch.from_numpy(mask_data).float().unsqueeze(0)
            labels = {k: torch.tensor(v) for k, v in labels.items()}
            # labels["centers"] = centers  # 保留原始坐标用于可视化

            return patch_tensor, mask_tensor, labels

        except Exception as e:
            logger.error(f"处理病例 {case['case_id']} 时出错: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))  # 跳过错误样本
        
    def _generate_labels(self, patch_shape, centers, mask_patch):
        max_queries = self.max_queries
        labels = {
            "exist": np.zeros(max_queries, dtype=np.int64),
            "sphere_coords": np.zeros((max_queries, 4)),
            "edges": np.zeros((max_queries, 6)),
            "orig_size": np.array(patch_shape),
            "labels": np.zeros(max_queries, dtype=np.int64),
            "has_edges": np.zeros((max_queries, 6))
        }

        valid_labels = []
        for idx, (label, x, y, z) in enumerate(centers[:max_queries]):
            if self.vertebra_range[0] <= label <= self.vertebra_range[1]:
                # 使用实际尺寸归一化
                h, w, d = patch_shape
                cx = x / h
                cy = y / w
                cz = z / d
                
                # 半径计算（考虑各向异性）
                radius = self._calc_radius(mask_patch, (x,y,z)) / np.sqrt(h**2 + w**2 + d**2)
                
                labels["exist"][label] = 1
                labels["labels"][idx] = label
                labels["sphere_coords"][label] = [cx, cy, cz, radius]
                valid_labels.append(label)

        # 生成边缘连接
        valid_labels.sort()
        for i in range(len(valid_labels)):
            current = valid_labels[i]
            if i > 0:
                prev = valid_labels[i-1]
                displacement = labels["sphere_coords"][current][:3] - labels["sphere_coords"][prev][:3]
                labels["edges"][current, :3] = displacement
                labels["has_edges"][current, :3] = 1
            if i < len(valid_labels)-1:
                next_ = valid_labels[i+1]
                displacement = labels["sphere_coords"][next_][:3] - labels["sphere_coords"][current][:3]
                labels["edges"][current, 3:] = displacement
                labels["has_edges"][current, 3:] = 1

        return labels
    
    def _calc_radius(self, mask, center):
        """改进的半径计算方法"""
        x, y, z = map(int, center)
        if not (0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= z < mask.shape[2]):
            return 0.0

        label = mask[x, y, z]
        if label == 0:
            return 0.0

        # 创建标签mask
        label_mask = (mask == label)
        
        # 计算距离场
        distance_field = ndimage.distance_transform_edt(label_mask)
        
        # 获取最大内切球半径
        return distance_field[x, y, z]


if __name__ == "__main__":
    # 初始化数据集
    dataset = SpineCTDataset(
        rawdata_dir=Path("data/train/rawdata"),
        derivatives_dir=Path("data/train/derivatives"),
        patch_size=192,
        resample_spacing=1.0
    )

    data_loader = DataLoader(dataset, batch_size=1,
                            num_workers=4, pin_memory=True, shuffle=True)\
                            
    for patch_tensor, mask_tensor, targets in data_loader:
        print(f'patch_tensor.shape: {patch_tensor.shape}')
        print(f'mask_tensor.shape: {mask_tensor.shape}')
        print(f'targets.shape: {targets.shape}')



    ## 可视化
    # patch_tensor, mask_tensor, labels = dataset[0]

    # ct_data = patch_tensor.squeeze().numpy() # 去除通道维度，得到[H,W,D]
    # mask_data = mask_tensor.squeeze().numpy()

    # im_np_sag = ct_data[:, :, int(ct_data.shape[2]/2)] # 获取中间切片
    # im_np_cor = ct_data[:, int(ct_data.shape[1]/2), :]

    # msk_np_sag = mask_data[:, :, int(mask_data.shape[2]/2)]
    # msk_np_sag[msk_np_sag == 0] = np.nan

    # msk_np_cor = mask_data[:, int(mask_data.shape[1]/2), :]
    # msk_np_cor[msk_np_cor == 0] = np.nan

    # result_dir = "result"
    # os.makedirs(result_dir, exist_ok=True)


    # plt.imshow(im_np_sag, cmap='gray', vmin=-1000, vmax=1500) # CT 图像
    # plt.imshow(msk_np_sag, cmap='jet', alpha=0.3, vmin=1, vmax=64) # 分割掩码
    # plt.axis('off')
    # plt.savefig(os.path.join(result_dir, "sagittal_view.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.close()

    # plt.imshow(im_np_cor, cmap='gray', vmin=-1000, vmax=1500)
    # plt.imshow(msk_np_cor, cmap='jet', alpha=0.3, vmin=1, vmax=64)
    # plt.axis('off')
    # plt.savefig(os.path.join(result_dir, "coronal_view.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.close()

    # checkpoint_path = "./checkpoints/model_epoch_30_valloss_2.04.pth"


