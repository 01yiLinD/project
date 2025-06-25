import os
import zipfile
import tarfile
import gzip
import shutil
from rarfile import RarFile


def extract_all_in_directory(directory):
    """
    批量解压目录中的所有压缩文件到当前文件夹。
    :param directory: 包含压缩文件的目录路径
    """
    if os.listdir(directory):
        print('Yes')
    else:
        print('No')

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".zip"):
                    # 解压 .zip 文件
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(root)
                        print(f"已解压 ZIP 文件: {file_path}")

                elif file.endswith((".tar.gz", ".tgz", ".tar")):
                    # 解压 .tar.gz 或 .tar 文件
                    with tarfile.open(file_path, 'r:*') as tar_ref:
                        tar_ref.extractall(root)
                        print(f"已解压 TAR 文件: {file_path}")

                elif file.endswith(".gz") and not file.endswith(".tar.gz"):
                    # 解压 .gz 文件（单个文件）
                    print(file)
                    output_file = os.path.join(root, file[:-3])  # 去掉 .gz 后缀
                    with gzip.open(file_path, 'rb') as f_in:
                        with open(output_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"已解压 GZ 文件: {file_path} 到 {output_file}")

                elif file.endswith(".rar"):
                    # 解压 .rar 文件
                    with RarFile(file_path) as rar_ref:
                        rar_ref.extractall(root)
                        print(f"已解压 RAR 文件: {file_path}")

                else:
                    print(f"不支持的文件格式: {file_path}")

            except Exception as e:
                print(f"解压失败: {file_path}, 错误: {e}")


# 示例用法
directory = "../data/train/derivatives"  # 替换为你的目录路径
extract_all_in_directory(directory)