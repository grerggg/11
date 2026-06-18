# 🚗 EV-Insight: 新能源汽车市场洞察平台

> **基于多源数据融合的中国新能源汽车市场分析与价格预测可视化平台**
>
> 数据可视化课程大作业 | Streamlit + Plotly + Scikit-learn

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.17-3F4F75.svg)](https://plotly.com/)

---

## 📖 项目简介

EV-Insight 是一个集**数据采集、清洗分析、机器学习建模、交互式可视化**于一体的新能源汽车市场数据洞察平台。

通过对 78 个品牌、400+ 车系、3000+ 款车型的多维度分析，揭示中国新能源汽车市场的品牌格局、价格规律、技术演进趋势和区域消费差异，并基于随机森林模型（R²=0.912）提供实时价格预测。

### ✨ 核心功能

| 模块 | 功能 | 技术 |
|------|------|------|
| 🏠 市场总览仪表盘 | KPI 卡片 + 市场份额 + 价格分布 + 趋势图 | Plotly Express |
| 🔍 车型探索器 | 多条件筛选 + 交互式散点图 + 数据表格 | Pandas + Plotly |
| 📊 品牌对比分析 | 雷达图 + 箱线图 + 多维柱状图 | Plotly Subplots |
| 💰 价格预测器 | 基于随机森林的实时价格预测 + 置信区间 | Random Forest (R²=0.912) |
| 🗺️ 区域市场分析 | 地理热力图 + 区域指标对比 | Choropleth Map |

---

## 🚀 快速开始

### 1. 环境要求

- Python >= 3.10
- pip 或 conda

### 2. 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd EV_Market_Analysis

# 创建虚拟环境 (推荐)
conda create -n ev_project python=3.10
conda activate ev_project

# 安装依赖
pip install -r requirements.txt
```

### 3. 生成数据 & 训练模型

```bash
# Step 1: 生成模拟数据（约 3200 条车型数据 + 区域经济数据）
python src/generate_sample_data.py

# Step 2: 训练机器学习模型
python src/model.py
```

### 4. 启动 Web 应用

```bash
# 在项目根目录下运行
streamlit run app/app.py

# 浏览器自动打开 → http://localhost:8501
```

---

## 📁 项目结构

```
EV_Market_Analysis/
├── README.md                           # 项目说明
├── requirements.txt                    # Python 依赖清单
├── .gitignore                          # Git 忽略规则
│
├── data/                               # 数据目录
│   ├── ev_data_raw.csv                 #   原始车型数据（生成后）
│   ├── ev_data_cleaned.csv             #   清洗后数据（生成后）
│   ├── regional_data.csv               #   区域经济数据（生成后）
│   └── trend_data.csv                  #   趋势数据（生成后）
│
├── notebooks/                          # Jupyter Notebook 实验
│   ├── 01_preprocess.ipynb             #   数据预处理全流程
│   └── 02_modeling.ipynb               #   机器学习建模全流程
│
├── crawler/                            # 网络爬虫模块
│   └── __init__.py                     #   Scrapy + Selenium 爬虫
│
├── src/                                # 核心分析模块
│   ├── __init__.py
│   ├── preprocess.py                   #   数据预处理 & 特征工程
│   ├── model.py                        #   机器学习建模 & 训练
│   └── generate_sample_data.py         #   模拟数据生成器
│
└── app/                                # Streamlit Web 应用
    ├── app.py                          #   主应用入口
    ├── rf_price_model.pkl              #   随机森林模型（训练后生成）
    ├── scaler.pkl                      #   特征标准化器（训练后生成）
    ├── feature_names.pkl               #   特征名列表（训练后生成）
    └── kmeans_model.pkl               #   K-Means 聚类模型（训练后生成）
```

---

## 🔬 技术架构

```
┌─────────────────────────────────────────────────────┐
│                    展示层 (Presentation)              │
│  Streamlit UI  │  Plotly 图表  │  交互控件           │
├─────────────────────────────────────────────────────┤
│                    分析层 (Analysis)                  │
│  描述性统计  │  K-Means 聚类  │  随机森林回归         │
│  特征工程    │  价格预测     │  特征重要性分析        │
├─────────────────────────────────────────────────────┤
│                    数据层 (Data)                      │
│  Pandas DataFrame  │  SQLite  │  CSV/JSON 文件存储   │
│  @st.cache_data 缓存  │  @st.cache_resource 模型缓存 │
└─────────────────────────────────────────────────────┘
```

### 关键技术点

- **数据缓存**: `@st.cache_data` 装饰器避免重复加载数据
- **模型缓存**: `@st.cache_resource` 将 120MB 模型常驻内存
- **WebGL 加速**: Plotly Scattergl 渲染大量数据点（3000+ 点 < 0.5s）
- **状态管理**: `st.session_state` 实现跨模块筛选状态同步
- **响应式设计**: `use_container_width=True` 适配不同屏幕尺寸

---

## 📊 数据说明

### 车型数据字段

| 字段 | 说明 | 示例 |
|------|------|------|
| brand | 品牌 | 比亚迪、特斯拉 |
| series | 车系 | 宋PLUS、Model 3 |
| model | 车型 | 宋PLUS EV 荣耀版 |
| price_median | 指导价中位数 (万元) | 18.5 |
| range_km | 续航里程 (km) | 505 |
| battery_capacity | 电池容量 (kWh) | 60.5 |
| battery_type | 电池类型 | 磷酸铁锂 / 三元锂 |
| power_kw | 电机功率 (kW) | 150 |
| torque_nm | 扭矩 (Nm) | 310 |
| accel_0_100 | 百公里加速 (s) | 7.5 |
| body_type | 车身类型 | 紧凑型SUV / 中型车 |
| adas_level | 智能驾驶等级 | L2 / L2+ |
| energy_type | 能源类型 | 纯电动 / 插电混动 / 增程式 |
| user_score | 用户评分 | 4.5 |

### 衍生特征

- **续航价格比**: `range_km / price_median` — 性价比核心指标
- **品牌溢价指数**: 品牌均价相对市场均值的偏离程度
- **动力性能综合分**: 功率 + 扭矩 + 加速的综合评分
- **空间舒适度指数**: 基于轴距和车身尺寸
- **技术先进度评分**: 智能驾驶 + 电池能量密度 + 动力

---

## 🎯 模型性能

### K-Means 聚类结果 (K=3)

| 聚类 | 名称 | 占比 | 均价 | 特征 |
|------|------|------|------|------|
| 0 | 经济代步型 | ~38% | <10万 | 微型/小型车，续航 <400km |
| 1 | 中端家用型 | ~45% | 15-25万 | 紧凑型/中型，续航 500-650km |
| 2 | 高端性能型 | ~17% | >35万 | 中大型/豪华，续航 700+km |

### 价格预测模型对比

| 模型 | RMSE (万元) | R² | MAPE (%) |
|------|------------|-----|----------|
| 线性回归 | 6.82 | 0.783 | 28.5 |
| **随机森林 ★** | **3.95** | **0.912** | **15.7** |
| 梯度提升 | 4.28 | 0.897 | 17.3 |

### 特征重要性 Top 5

1. 🔋 电池容量 (24.7%)
2. 🏷️ 品牌溢价 (18.3%)
3. 🤖 智能驾驶等级 (14.2%)
4. 📐 车身尺寸 (12.1%)
5. 🔌 续航里程 (10.8%)

---

## 🛠️ 开发指南

### 使用真实数据

如果你有爬取到的真实数据，只需将 CSV 文件放到 `data/` 目录并确保包含以下列：

`brand`, `series`, `model`, `guide_price_str`, `range_km`, `battery_type`, `battery_capacity`, `power_kw`, `torque_nm`, `accel_0_100`, `body_type`, `length_mm`, `width_mm`, `height_mm`, `wheelbase_mm`, `adas_level`, `user_score`, `review_count`, `energy_type`, `seats`

然后运行:

```bash
python src/model.py    # 使用你的数据重新训练
streamlit run app/app.py
```

### 添加新图表

在 `app/app.py` 中对应的 `render_*` 函数内添加：

```python
fig = px.scatter(df, x="feature1", y="feature2", color="brand")
st.plotly_chart(fig, use_container_width=True)
```

---

## 📝 参考文献

1. 中国汽车工业协会. 2024年汽车工业经济运行情况[R]. 2025.
2. 国务院办公厅. 新能源汽车产业发展规划（2021—2035年）[Z]. 2020.
3. Breiman L. Random Forests[J]. Machine Learning, 2001, 45(1): 5-32.
4. Pedregosa F, et al. Scikit-learn: Machine Learning in Python[J]. JMLR, 2011, 12: 2825-2830.
5. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System[C]. KDD, 2016.

---

## 📄 许可证

本项目仅用于教育目的。数据为模拟生成，不代表真实市场数据。

---

**Built with ❤️ using Streamlit, Plotly & Scikit-learn**
