# Debug Note: CUDA Index Out of Bounds Error

## 问题表现

### 症状
训练时在特定 batch 出现 CUDA 错误：
```
RuntimeError: CUDA error: device-side assert triggered
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:94:
Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
```

### 初步观察
- 在 validation 的第 249 个 batch 稳定复现
- 前 248 个 batch 正常运行
- 错误发生在 embedding lookup 阶段

### 关键数据
```python
input_sequence.dtype = torch.int32  # 类型正确
input_sequence.shape = torch.Size([64, 256])  # 形状正确
input_sequence.max() = tensor(1953656678, dtype=torch.int32)  # ⚠️ 远超 vocab_size=10000
input_sequence.min() = tensor(0)  # 正常
```

---

## 排查过程

### 假设 1: 数据类型转换问题 ❌
**猜测**: `torch.LongTensor()` 导致 int32 → int64 的类型转换错误

**验证**:
```python
# 修改前
return torch.LongTensor(sequence), torch.LongTensor(target)

# 修改后
return torch.from_numpy(sequence), torch.from_numpy(target)
```

**结果**: 问题依旧，排除类型转换问题

---

### 假设 2: DataLoader 多进程 + memmap 冲突 ❌
**猜测**: `num_workers > 0` 导致子进程继承的 memmap 对象内存映射失效

**验证**:
```python
# 修改 DataLoader 配置
num_workers = 0  # 从 1 改为 0
persistent_workers = False
```

**结果**: 问题依旧，排除多进程问题

---

### 假设 3: 原始数据文件损坏 ❌
**猜测**: `.npy` 文件中存储的 token IDs 超出范围

**验证**:
```python
data = np.load('data/TinyStoriesV2-GPT4-valid_tinyTokenizer_10000.npy')
print(f'Max: {data.max()}')  # 9999 ✓
print(f'Min: {data.min()}')  # 0 ✓
print(f'First 10: {data[:10]}')  # [85 869 500 507 266 324 617 372 263 917] ✓
```

**结果**: 原始文件完全正常，最大值 9999 < vocab_size=10000

---

### 关键突破: 对比 np.load() vs np.memmap()

**测试代码**:
```python
# 方式 1: np.load()
data_loaded = np.load('data/xxx.npy')
print(f'Shape: {data_loaded.shape}')  # (5402838,)
print(f'Max: {data_loaded.max()}')    # 9999 ✓
print(f'First 10: {data_loaded[:10]}')  # [85 869 500 507 ...] ✓

# 方式 2: np.memmap()
data_memmap = np.memmap('data/xxx.npy', dtype=np.int32, mode='r')
print(f'Shape: {data_memmap.shape}')  # (5402870,) ⚠️ 多了 32 个元素
print(f'Max: {data_memmap.max()}')    # 1953656678 ❌
print(f'First 10: {data_memmap[:10]}')  # [1297436307 88400 ...] ❌ 完全错误
```

**发现**:
1. **Shape 不同**: memmap 多了 32 个元素 (5402870 vs 5402838)
2. **数据不同**: memmap 的前面数据是乱码，包含 1953656678 等巨大数值
3. **差值**: 5402870 - 5402838 = 32 个 int32 = 128 字节

---

## 根本原因

### `.npy` 文件格式
`.npy` 文件包含：
- **Header (128 bytes)**: 存储元数据 (magic number, version, dtype, shape 等)
- **Data**: 实际的 numpy array 数据

### `np.memmap()` 的行为
`np.memmap(path, dtype=np.int32, mode='r')` **不会解析 `.npy` 的 header**，而是：
1. 直接把整个文件当作原始二进制数据
2. 从文件开头按 int32 读取
3. 结果：**前 128 字节的 header 被误读为 32 个 int32 数值**

### 为什么出现 1953656678？
```
Header 的某 4 个字节: 0x74 0x72 0x61 0x4e (ASCII: "traN", 可能是 "numpy" 的一部分)
按 little-endian int32 解析: 0x4e617274 = 1314145908
其他组合产生: 1953656678 等看起来"随机"的大数
```

这些 header 字节被错误解析为 token IDs，远超 vocab_size=10000，导致 embedding lookup 越界。

---

## 解决方案

### 修改前
```python
def build_torch_dataset(self, data_split: str) -> TokenDataset:
    data_path = self.config['data'][f'{data_split}_dataset']
    data = np.memmap(data_path, dtype=np.int32, mode='r')  # ❌ 跳过 header
    return TokenDataset(tokens=data, context_length=self.config['model']['context_length'])
```

### 修改后
```python
def build_torch_dataset(self, data_split: str) -> TokenDataset:
    data_path = self.config['data'][f'{data_split}_dataset']
    data = np.load(file=data_path, mmap_mode='r')  # ✓ 正确解析 header + 内存映射
    return TokenDataset(tokens=data, context_length=self.config['model']['context_length'])
```

### 验证
```python
dataset = TokenDataset(tokens=data, context_length=256)

# 测试之前有问题的 Sample 0
seq, target = dataset[0]
print(f'Sample 0: max={seq.max().item()}')  # 4358 ✓ (之前是 1953656678)

# 测试所有关键样本
for idx in [0, 100, 248, 249, 250, len(dataset)-1]:
    seq, target = dataset[idx]
    assert seq.max() < 10000  # ✓ 全部通过
```

---

## 关键学习点

### 1. `np.load()` vs `np.memmap()` 的区别

| 方法 | 适用场景 | Header 处理 | 内存占用 |
|------|---------|------------|---------|
| `np.load(path)` | `.npy` 文件 | ✓ 自动解析 | 加载全部到内存 |
| `np.load(path, mmap_mode='r')` | `.npy` 文件 | ✓ 自动解析 | 内存映射 (推荐) |
| `np.memmap(path, dtype, mode)` | 原始二进制文件 | ✗ 不处理 | 内存映射 |

### 2. 正确的使用场景

```python
# ✓ 对于 .npy 文件 (有 header)
data = np.load('file.npy', mmap_mode='r')

# ✓ 对于原始二进制文件 (无 header，如 .bin, .dat)
data = np.memmap('file.bin', dtype=np.int32, mode='r')

# ✓ 如果想用 memmap，应该保存为原始格式
array.tofile('file.bin')  # 保存时
data = np.memmap('file.bin', dtype=np.int32, mode='r')  # 读取时
```

### 3. 调试技巧

当遇到"数据异常"问题时：
1. **对比不同加载方式**: np.load() vs np.memmap()
2. **检查 shape**: 差异通常暴露问题 (如多 32 个元素 → header)
3. **检查前几个元素**: 是否是预期值？
4. **检查异常值的模式**: 随机大数 → 可能是二进制误读
5. **启用 CUDA_LAUNCH_BLOCKING=1**: 获取准确的错误位置

---

## 总结

- **表现**: CUDA embedding lookup index out of bounds
- **表象原因**: token ID = 1953656678 >> vocab_size = 10000
- **深层原因**: `np.memmap()` 把 `.npy` 的 128 字节 header 误读为数据
- **解决方案**: 使用 `np.load(mmap_mode='r')` 代替 `np.memmap()`
- **教训**: 理解文件格式和工具的适用场景很重要

---

**Date**: 2025-10-18
**Author**: Debugging session with Claude
