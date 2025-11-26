#!/bin/bash

# 遗传算法训练人形机器人 - 快速开始脚本

echo "========================================"
echo "遗传算法训练3D人形机器人 - 快速开始"
echo "========================================"
echo ""

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"
echo ""

# 创建虚拟环境（推荐）
read -p "是否创建Python虚拟环境? (推荐) [Y/n]: " create_venv
create_venv=${create_venv:-Y}

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ 虚拟环境已激活"
    echo ""
fi

# 安装依赖
echo "安装依赖包..."
echo "这可能需要几分钟..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ 依赖安装成功"
else
    echo "✗ 依赖安装失败"
    echo "请检查错误信息并手动安装"
    exit 1
fi

echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "现在您可以："
echo "  1. 开始训练: python train_humanoid.py"
echo "  2. 可视化结果: python visualize.py"
echo ""
echo "如果使用了虚拟环境，下次使用前请先激活："
echo "  source venv/bin/activate"
echo ""

