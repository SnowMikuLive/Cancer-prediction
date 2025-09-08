# 癌症诊断分类：K近邻分类器

**作者**: C.C.
**项目地址**：https://github.com/SnowMikuLive/Cancer-prediction

使用威斯康星州乳腺肿瘤数据集训练K近邻分类器来预测肿瘤是恶性还是良性。

## 📋 项目概述

本项目实现了一个基于K近邻算法的乳腺癌诊断分类系统，能够根据肿瘤的30个数值特征准确预测肿瘤是恶性还是良性。该系统在威斯康星州乳腺肿瘤数据集上达到了优秀的分类性能。

## 🎯 项目目标

- 使用机器学习技术进行癌症诊断辅助
- 实现高精度的肿瘤分类预测
- 提供完整的模型训练和评估流程
- 生成详细的可视化分析报告

## 📊 数据集信息

**数据集**: 威斯康星州乳腺肿瘤数据集 (Wisconsin Breast Cancer Dataset)

- **样本数量**: 569个
- **特征数量**: 30个数值特征
- **目标变量**: 诊断结果 (B=良性, M=恶性)
- **数据分布**: 
  - 良性样本: 357个 (62.7%)
  - 恶性样本: 212个 (37.3%)

### 主要特征类别

1. **半径特征** (radius): 从中心到边界点的平均距离
2. **纹理特征** (texture): 灰度值的标准偏差
3. **周长特征** (perimeter): 肿瘤周长
4. **面积特征** (area): 肿瘤面积
5. **平滑度特征** (smoothness): 半径长度的局部变化
6. **紧密度特征** (compactness): 周长²/面积 - 1.0
7. **凹度特征** (concavity): 轮廓凹部分的严重程度
8. **凹点特征** (concave points): 轮廓凹部分的数量
9. **对称性特征** (symmetry): 肿瘤的对称性
10. **分形维数** (fractal dimension): 海岸线近似 - 1

每个特征都有三个统计量：均值(mean)、标准误差(se)、最差值(worst)

## 🛠️ 技术栈

- **Python 3.x**
- **pandas**: 数据处理和分析
- **numpy**: 数值计算
- **matplotlib**: 基础绘图
- **seaborn**: 统计可视化
- **scikit-learn**: 机器学习算法和工具
- **Jupyter Notebook**: 交互式开发环境

## 📁 项目结构

```
Cancer prediction/
├── README.md                           # 项目说明文档
├── cancer_prediction_knn.ipynb        # 主要分析notebook
├── bc_data.csv                        # 威斯康星州乳腺肿瘤数据集
├── data_visualization.png             # 数据探索可视化
├── model_results_visualization.png    # 模型结果可视化
├── k_value_selection.png              # K值选择曲线
├── confusion_matrix.png               # 混淆矩阵
├── roc_curve.png                      # ROC曲线
├── performance_metrics.png            # 性能指标柱状图
├── tumor_type_distribution.png        # 肿瘤类型分布
├── feature_distribution_boxplot.png   # 特征分布箱线图
├── feature_correlation_heatmap.png    # 特征相关性热图
└── sample_count_comparison.png        # 样本数量对比
```

## 🚀 快速开始

### 环境要求

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 运行步骤

1. **克隆或下载项目**
   ```bash
   git clone <repository-url>
   cd "Cancer prediction"
   ```

2. **启动Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

3. **打开并运行notebook**
   - 打开 `cancer_prediction_knn.ipynb`
   - 按顺序运行所有代码单元格
   - 查看生成的图表和结果

## 📈 模型性能

### 最佳参数
- **最佳K值**: 通过网格搜索确定
- **交叉验证**: 5折交叉验证
- **特征标准化**: StandardScaler

### 性能指标
- **准确率 (Accuracy)**: >95%
- **精确率 (Precision)**: >95%
- **召回率 (Recall)**: >95%
- **特异性 (Specificity)**: >95%
- **F1分数**: >95%
- **ROC AUC**: >0.95

## 📊 可视化内容

### 数据探索可视化
1. **肿瘤类型分布饼图**: 显示良性和恶性样本的比例
2. **特征分布箱线图**: 对比不同类别样本的特征分布
3. **特征相关性热图**: 展示特征间的相关关系
4. **样本数量对比图**: 直观显示样本分布

### 模型结果可视化
1. **K值选择曲线**: 展示不同K值下的模型性能
2. **混淆矩阵**: 详细显示分类结果
3. **ROC曲线**: 评估分类器性能
4. **性能指标柱状图**: 对比各项评估指标

## 🔧 核心功能

### 1. 数据预处理
- 数据加载和探索
- 缺失值检查
- 特征标准化
- 训练/测试集分割

### 2. 模型训练
- K近邻分类器训练
- 网格搜索参数优化
- 交叉验证性能评估

### 3. 模型评估
- 多种性能指标计算
- 混淆矩阵分析
- ROC曲线绘制
- 分类报告生成

### 4. 结果可视化
- 自动生成高质量图表
- 保存为PNG格式 (300 DPI)
- 支持中文标签显示

## 📝 使用示例

```python
# 预测新样本
def predict_tumor_type(features):
    """
    预测肿瘤类型
    输入: 30维特征向量
    输出: 预测结果和概率
    """
    features_scaled = scaler.transform([features])
    prediction = best_knn.predict(features_scaled)[0]
    probability = best_knn.predict_proba(features_scaled)[0]
    
    return prediction, probability

# 使用示例
sample_features = [17.99, 10.38, 122.8, 1001, 0.1184, ...]  # 30个特征值
prediction, prob = predict_tumor_type(sample_features)
print(f"预测结果: {'恶性' if prediction == 1 else '良性'}")
print(f"预测概率: 良性={prob[0]:.3f}, 恶性={prob[1]:.3f}")
```

## 📊 结果解读

### 混淆矩阵解读
- **真阴性 (TN)**: 正确预测为良性的样本数
- **假阳性 (FP)**: 错误预测为恶性的良性样本数
- **假阴性 (FN)**: 错误预测为良性的恶性样本数
- **真阳性 (TP)**: 正确预测为恶性的样本数

### ROC曲线解读
- **AUC值**: 越接近1.0表示分类性能越好
- **曲线位置**: 越靠近左上角性能越好
- **对角线**: 表示随机分类器的性能

## 🔍 模型特点

### 优势
- **简单易懂**: K近邻算法原理简单，易于理解
- **无需训练**: 基于实例的学习，无需复杂的训练过程
- **高准确率**: 在乳腺癌数据集上表现优秀
- **可解释性**: 可以分析最近邻样本进行解释

### 局限性
- **计算复杂度**: 预测时需要计算与所有训练样本的距离
- **特征敏感**: 对特征缩放和维度敏感
- **内存需求**: 需要存储所有训练样本

## 🎯 应用场景

- **医疗诊断辅助**: 为医生提供初步诊断参考
- **医学研究**: 分析肿瘤特征与恶性程度的关系
- **教学演示**: 机器学习在医疗领域的应用示例
- **算法比较**: 作为其他分类算法的基准

## 📚 参考文献

1. Wolberg, W.H., & Mangasarian, O.L. (1990). Multisurface method of pattern separation for medical diagnosis applied to breast cytology. Proceedings of the National Academy of Sciences, 87(23), 9193-9196.

2. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

## 📄 许可证

本项目仅用于学习和研究目的。请勿将结果用于实际医疗诊断。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 联系方式

**项目作者**: C.C.
**项目地址**：https://github.com/SnowMikuLive/Cancer-prediction
**免责声明**: 本项目仅用于教育和研究目的，不能替代专业医疗诊断。任何医疗决策都应咨询专业医生。
