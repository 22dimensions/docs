import os
import torch
import torch.nn.functional as F

def compare_cosine_similarity(dir1, dir2):
    # 获取目录下的所有文件（排除子目录）
    files1 = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
    files2 = [f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]
    common_files = set(files1).intersection(set(files2))

    for file in common_files:
        path1 = os.path.join(dir1, file)
        path2 = os.path.join(dir2, file)

        try:
            # 加载文件到CPU，避免设备不兼容
            data1 = torch.load(path1, map_location='cpu')
            data2 = torch.load(path2, map_location='cpu')
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        # data1 = data1.squeeze(0)

        # data1 = data1.transpose(0, 1)

        # data1 = data1.reshape(5, -1)


        # 处理状态字典（若文件保存的是模型参数）
        def extract_tensor(data):
            if isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, dict):  # 状态字典
                return torch.cat([v.flatten() for v in data.values()])
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
        # 检查形状是否相同
        if data1.shape != data2.shape:
            print(f"Shapes of {file} do not match. Skipping.")
            print("vec1 shape", data1.shape)
            print("vec2 shape", data2.shape)
            continue

        try:
            vec1 = extract_tensor(data1).flatten()
            vec2 = extract_tensor(data2).flatten()
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

        

        # 计算余弦相似度
        similarity = F.cosine_similarity(vec1, vec2, dim=0)
        print("------------------new data line ----------------------")
        print("no_quant data: ", data1[-15:-1])
        print("fa3 data: ", data2[-15:-1])
        print(f"{file} shape: {data1.shape} similarity: {similarity.item():.4f} torch.equal: {torch.equal(data1, data2)}")

# 使用示例
dir1 = "/root/dump_data/ms_fa3"
dir2 = "/root/dump_data/fa3_manual"
compare_cosine_similarity(dir1, dir2)
