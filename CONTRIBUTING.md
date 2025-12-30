# Contributing to BWGNN-ResAFA

感谢您对 BWGNN-ResAFA 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

如果您发现了 bug 或有功能建议，请通过 GitHub Issues 提交：

1. 检查是否已有相关 Issue
2. 如果没有，创建新的 Issue
3. 提供清晰的问题描述、复现步骤和预期行为

### 提交代码

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 遵循 PEP 8 Python 代码风格指南
- 添加适当的注释和文档字符串
- 确保代码通过基本的语法检查
- 如果可能，添加单元测试

### 实验贡献

如果您有新的实验想法或结果：

1. 在 `Experiment_Log.md` 中记录实验过程
2. 更新 `README_Experiment.md` 添加新结果
3. 确保实验结果可复现
4. 提供训练命令和超参数设置

## 开发环境设置

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/BWGNN-ResAFA.git
cd BWGNN-ResAFA
```

2. 创建虚拟环境：
```bash
conda create -n eghrn python=3.8 -y
conda activate eghrn
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 测试

在提交代码前，请确保：

- 代码可以正常运行
- 没有引入新的错误
- 文档已更新

## 问题反馈

如有任何问题，请通过 GitHub Issues 联系我们。

感谢您的贡献！

