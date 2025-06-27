# 贡献指南

感谢您对CBIR系统的关注！我们欢迎所有形式的贡献，包括但不限于：

- 🐛 Bug报告
- 💡 功能建议
- 📝 文档改进
- 🔧 代码优化
- 🧪 测试用例

## 如何贡献

### 1. Fork和Clone

1. Fork本仓库到您的GitHub账户
2. Clone您的fork到本地：
```bash
git clone https://github.com/your-username/cbir-system.git
cd cbir-system
```

### 2. 创建分支

为您的贡献创建一个新分支：
```bash
git checkout -b feature/your-feature-name
# 或者
git checkout -b fix/your-bug-fix
```

### 3. 开发

- 遵循现有的代码风格
- 添加必要的注释和文档
- 确保代码通过所有测试

### 4. 提交

提交您的更改：
```bash
git add .
git commit -m "feat: add new feature description"
```

### 5. 推送和Pull Request

```bash
git push origin feature/your-feature-name
```

然后在GitHub上创建Pull Request。

## 代码规范

### Python代码风格

- 遵循PEP 8规范
- 使用4个空格缩进
- 行长度不超过127字符
- 使用有意义的变量和函数名

### 提交信息格式

使用[Conventional Commits](https://www.conventionalcommits.org/)格式：

```
type(scope): description

[optional body]

[optional footer]
```

类型包括：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### 示例

```
feat(web): add image upload progress bar

- Add progress indicator for large file uploads
- Improve user experience during image processing

Closes #123
```

## 报告Bug

如果您发现了Bug，请：

1. 检查是否已有相关Issue
2. 创建新的Issue，包含：
   - Bug的详细描述
   - 重现步骤
   - 期望行为
   - 实际行为
   - 环境信息（操作系统、Python版本等）

## 功能建议

如果您有新功能建议，请：

1. 检查是否已有相关讨论
2. 创建Issue描述您的想法
3. 说明功能的价值和实现思路

## 开发环境设置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 安装开发依赖：
```bash
pip install -r requirements-dev.txt
```

3. 运行测试：
```bash
pytest
```

4. 代码检查：
```bash
flake8 src/
```

## 许可证

通过提交Pull Request，您同意您的贡献将在MIT许可证下发布。

## 联系方式

如果您有任何问题，请：

- 创建GitHub Issue
- 发送邮件到项目维护者
- 参与项目讨论

感谢您的贡献！🎉 