import torch


def check_cuda():
    # 基础检查
    cuda_available = torch.cuda.is_available()
    print(f"1. CUDA 可用性: {'✅可用' if cuda_available else '❌不可用'}")

    if not cuda_available:
        return

    # 设备信息
    device_count = torch.cuda.device_count()
    print(f"2. 检测到 {device_count} 个 GPU 设备")

    for i in range(device_count):
        print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"     计算能力: {torch.cuda.get_device_capability(i)}")
        print(f"     显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")

    # 实际张量计算测试
    print("\n3. 运行计算测试...")
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = (x + y).sum()
        print(f"   计算测试结果: {z.item()} (预期接近 0)")
        print("   ✅ CUDA 计算正常")
    except Exception as e:
        print(f"   ❌ 计算失败: {str(e)}")


if __name__ == "__main__":
    check_cuda()