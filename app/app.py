"""
=============================================================================
EV-Insight: 新能源汽车市场洞察平台
=============================================================================
基于 Streamlit 的交互式 Web 应用。

功能模块：
  1. 🏠 市场总览仪表盘 - 关键指标卡片 + 市场份额 + 价格分布
  2. 🔍 车型探索器     - 多条件筛选 + 交互式散点图 + 数据表格
  3. 📊 品牌对比分析   - 雷达图 + 箱线图 + 柱状图多维对比
  4. 💰 价格预测器     - 基于随机森林的实时价格预测
  5. 🗺️ 区域市场分析   - 地理热力图 + 区域对比

启动方式:
    streamlit run app.py
    或
    cd app/ && streamlit run app.py
=============================================================================
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="EV-Insight | 新能源汽车市场洞察平台",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS 样式
# =============================================================================

st.markdown("""
<style>
    /* 标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-top: -10px;
    }
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        padding: 20px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 数据加载（缓存）
# =============================================================================

@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """
    加载清洗后的新能源车型数据。

    自动尝试多个路径，兼容各种启动方式。
    """
    # 构建候选路径列表
    paths = [
        PROJECT_ROOT / "data" / "ev_data_cleaned.csv",
        Path(__file__).parent.parent / "data" / "ev_data_cleaned.csv",
        Path("data") / "ev_data_cleaned.csv",
        Path("../data") / "ev_data_cleaned.csv",
        Path.cwd() / "data" / "ev_data_cleaned.csv",
        Path.cwd() / "EV_Market_Analysis" / "data" / "ev_data_cleaned.csv",
        Path.home() / "Desktop" / "EV_Market_Analysis" / "data" / "ev_data_cleaned.csv",
    ]

    # 去重并转为绝对路径
    seen = set()
    unique_paths = []
    for p in paths:
        try:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_paths.append(resolved)
        except Exception:
            pass

    for p in unique_paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                required_cols = ["brand", "price_median", "range_km"]
                if all(c in df.columns for c in required_cols):
                    return df
            except Exception:
                continue

    # 所有路径都找不到
    searched = "\n".join(f"  - {p}" for p in unique_paths)
    st.error(f"""
    ❌ **找不到数据文件 `ev_data_cleaned.csv`**

    已搜索以下路径：
    {searched}

    请先运行以下命令生成模拟数据：
    ```
    cd EV_Market_Analysis
    python src/generate_sample_data.py
    ```
    """)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_regional_data() -> pd.DataFrame:
    """加载区域经济数据"""
    return _load_csv("regional_data.csv")


@st.cache_data(ttl=3600)
def load_trend_data() -> pd.DataFrame:
    """加载趋势数据"""
    return _load_csv("trend_data.csv")


def _load_csv(filename: str) -> pd.DataFrame:
    """通用 CSV 加载，自动搜索多个路径"""
    paths = [
        PROJECT_ROOT / "data" / filename,
        Path(__file__).parent.parent / "data" / filename,
        Path("data") / filename,
        Path("../data") / filename,
        Path.cwd() / "data" / filename,
    ]
    for p in paths:
        try:
            resolved = p.resolve()
            if resolved.exists():
                return pd.read_csv(resolved)
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_resource
def load_models():
    """
    加载训练好的机器学习模型。

    Returns
    -------
    dict 或 None
    """
    model_dir = Path(__file__).parent

    try:
        rf_model = joblib.load(model_dir / "rf_price_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        features = joblib.load(model_dir / "feature_names.pkl")
        kmeans = joblib.load(model_dir / "kmeans_model.pkl")

        # 尝试加载元数据
        metadata_path = model_dir / "model_metadata.pkl"
        metadata = None
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)

        return {
            "rf_model": rf_model,
            "scaler": scaler,
            "features": features,
            "kmeans": kmeans,
            "metadata": metadata,
            "loaded": True,
        }
    except FileNotFoundError:
        return {"loaded": False}
    except Exception as e:
        st.warning(f"模型加载失败: {e}")
        return {"loaded": False}


# =============================================================================
# 辅助组件
# =============================================================================


def render_kpi_cards(df: pd.DataFrame):
    """渲染关键指标卡片行"""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🚗 车型总数", f"{len(df):,}",
                  delta=None, delta_color="off")
    with col2:
        st.metric("🏷️ 品牌数", f"{df['brand'].nunique()}",
                  delta=None, delta_color="off")
    with col3:
        st.metric("💰 平均价格", f"{df['price_median'].mean():.1f} 万",
                  delta=None, delta_color="off")
    with col4:
        st.metric("🔋 平均续航", f"{df['range_km'].mean():.0f} km",
                  delta=None, delta_color="off")
    with col5:
        st.metric("⭐ 平均评分", f"{df['user_score'].mean():.1f}",
                  delta=None, delta_color="off")


# =============================================================================
# 模块 1: 市场总览仪表盘
# =============================================================================


def render_overview(df: pd.DataFrame):
    """渲染市场总览仪表盘"""
    st.header("🏠 市场总览仪表盘")
    st.markdown("---")

    # KPI 卡片
    render_kpi_cards(df)

    st.markdown("---")

    # 第一行图表
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📊 品牌市场份额 (Top 15)")

        # 品牌车型数量统计
        brand_counts = df["brand"].value_counts().head(15).reset_index()
        brand_counts.columns = ["品牌", "车型数"]

        fig = px.bar(
            brand_counts,
            x="车型数",
            y="品牌",
            orientation="h",
            color="车型数",
            color_continuous_scale="Viridis",
            text="车型数",
        )
        fig.update_layout(
            height=450,
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("💲 价格分布")

        fig = px.histogram(
            df,
            x="price_median",
            nbins=50,
            color_discrete_sequence=["#667eea"],
            marginal="box",
            labels={"price_median": "价格 (万元)", "count": "车型数"},
        )
        fig.add_vline(
            x=df["price_median"].median(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"中位数: {df['price_median'].median():.1f}万",
        )
        fig.update_layout(
            height=450,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 第二行图表
    col_left2, col_right2 = st.columns([1, 1])

    with col_left2:
        st.subheader("📈 续航里程 vs 价格 (气泡图)")

        # 采样避免渲染过多数据点
        sample_df = df.sample(min(1000, len(df)), random_state=42)

        fig = px.scatter(
            sample_df,
            x="range_km",
            y="price_median",
            size="battery_capacity",
            color="energy_type" if "energy_type" in df.columns else "brand",
            hover_name="model" if "model" in df.columns else None,
            hover_data={
                "brand": True,
                "range_km": True,
                "price_median": True,
                "battery_capacity": True,
            },
            labels={
                "range_km": "续航里程 (km)",
                "price_median": "价格 (万元)",
                "battery_capacity": "电池容量 (kWh)",
            },
            title="气泡大小 = 电池容量",
        )
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        st.subheader("🔋 续航里程趋势 (2020-2025)")

        trend_df = load_trend_data()
        if not trend_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df["year"],
                y=trend_df["avg_range_km"],
                mode="lines+markers",
                name="平均续航",
                line=dict(color="#667eea", width=3),
                marker=dict(size=10),
            ))
            # 添加范围带
            fig.add_trace(go.Scatter(
                x=trend_df["year"],
                y=trend_df["avg_range_km"] + trend_df["range_std_km"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=trend_df["year"],
                y=trend_df["avg_range_km"] - trend_df["range_std_km"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(102, 126, 234, 0.2)",
                name="±1 标准差",
            ))
            fig.update_layout(
                height=450,
                yaxis_title="续航里程 (km)",
                xaxis_title="年份",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("趋势数据未加载，请运行 generate_sample_data.py")

    # 价格等级分布
    st.markdown("---")
    st.subheader("📐 车型价格等级分布")

    if "price_category" in df.columns:
        cat_counts = df["price_category"].value_counts().reset_index()
        cat_counts.columns = ["价格等级", "车型数"]

        colors = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"]
        fig = px.pie(
            cat_counts,
            values="车型数",
            names="价格等级",
            color_discrete_sequence=colors,
            hole=0.4,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 回退：按价格区间手动分组
        bins = [0, 10, 20, 35, 60, float("inf")]
        labels = ["经济型(<10万)", "入门型(10-20万)", "中端型(20-35万)",
                  "高端型(35-60万)", "豪华型(>60万)"]
        df_temp = df.copy()
        df_temp["price_category"] = pd.cut(
            df_temp["price_median"], bins=bins, labels=labels
        )
        cat_counts = df_temp["price_category"].value_counts().sort_index().reset_index()
        cat_counts.columns = ["价格等级", "车型数"]

        colors = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"]
        fig = px.pie(
            cat_counts,
            values="车型数",
            names="价格等级",
            color_discrete_sequence=colors,
            hole=0.4,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 模块 2: 车型探索器
# =============================================================================


def render_vehicle_explorer(df: pd.DataFrame):
    """渲染车型探索器"""
    st.header("🔍 车型探索器")
    st.markdown("---")

    # ---- 筛选面板 ----
    with st.expander("⚙️ 筛选条件", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            all_brands = sorted(df["brand"].unique().tolist())
            selected_brands = st.multiselect(
                "品牌",
                options=all_brands,
                default=all_brands[:5],
            )

        with col2:
            price_min = float(df["price_median"].min())
            price_max = float(df["price_median"].max())
            price_range = st.slider(
                "价格区间 (万元)",
                min_value=price_min,
                max_value=price_max,
                value=(price_min, price_max),
                step=1.0,
            )

        with col3:
            range_min = int(df["range_km"].min())
            range_max = int(df["range_km"].max())
            range_filter = st.slider(
                "续航里程 (km)",
                min_value=range_min,
                max_value=range_max,
                value=(range_min, range_max),
                step=50,
            )

        with col4:
            energy_types = ["全部"]
            if "energy_type" in df.columns:
                energy_types += sorted(df["energy_type"].unique().tolist())
            selected_energy = st.selectbox("能源类型", options=energy_types)

        # 第二行筛选
        col5, col6 = st.columns(2)

        with col5:
            body_types = ["全部"]
            if "body_type" in df.columns:
                body_types += sorted(df["body_type"].unique().tolist())
            selected_body = st.selectbox("车身类型", options=body_types)

        with col6:
            adas_levels = ["全部"]
            if "adas_level" in df.columns:
                adas_levels += sorted(df["adas_level"].unique().tolist())
            selected_adas = st.selectbox("智能驾驶等级", options=adas_levels)

    # ---- 数据筛选 ----
    filtered_df = df.copy()
    if selected_brands:
        filtered_df = filtered_df[filtered_df["brand"].isin(selected_brands)]
    filtered_df = filtered_df[
        (filtered_df["price_median"] >= price_range[0])
        & (filtered_df["price_median"] <= price_range[1])
    ]
    filtered_df = filtered_df[
        (filtered_df["range_km"] >= range_filter[0])
        & (filtered_df["range_km"] <= range_filter[1])
    ]
    if selected_energy != "全部" and "energy_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["energy_type"] == selected_energy]
    if selected_body != "全部" and "body_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["body_type"] == selected_body]
    if selected_adas != "全部" and "adas_level" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["adas_level"] == selected_adas]

    st.markdown(f"**筛选结果: {len(filtered_df)} 款车型**")

    # ---- 交互式散点图 ----
    st.markdown("---")
    st.subheader("📈 自定义散点图探索")

    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    # 过滤掉编码列和其他不适合展示的列
    scatter_cols = [c for c in numeric_cols
                    if not c.endswith("_encoded") and c not in
                    ["price_low", "price_high", "seats"]]

    col_x, col_y, col_color = st.columns(3)
    with col_x:
        x_axis = st.selectbox(
            "X 轴",
            options=scatter_cols,
            index=scatter_cols.index("range_km") if "range_km" in scatter_cols else 0,
        )
    with col_y:
        y_axis = st.selectbox(
            "Y 轴",
            options=scatter_cols,
            index=scatter_cols.index("price_median") if "price_median" in scatter_cols else 0,
        )
    with col_color:
        color_cols = ["brand"]
        if "energy_type" in filtered_df.columns:
            color_cols.append("energy_type")
        if "body_type" in filtered_df.columns:
            color_cols.append("body_type")
        if "cluster_name" in filtered_df.columns:
            color_cols.append("cluster_name")
        color_by = st.selectbox("颜色分组", options=color_cols)

    # 如果数据量太大，采样展示
    plot_df = filtered_df
    if len(plot_df) > 1500:
        plot_df = plot_df.sample(1500, random_state=42)

    size_col = None
    if "battery_capacity" in scatter_cols:
        size_col = "battery_capacity"

    fig = px.scatter(
        plot_df,
        x=x_axis,
        y=y_axis,
        color=color_by,
        size=size_col,
        hover_name="model" if "model" in plot_df.columns else None,
        hover_data=["brand", "price_median", "range_km"],
        opacity=0.7,
    )
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # ---- 数据表格 ----
    st.markdown("---")
    st.subheader("📋 车型数据详情")

    display_cols = [
        "brand", "model", "price_median", "range_km", "battery_capacity",
        "power_kw", "accel_0_100", "adas_level", "body_type", "user_score",
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[display_cols]
        .sort_values("price_median")
        .style.format({
            "price_median": "{:.1f}万",
            "range_km": "{:.0f}km",
            "battery_capacity": "{:.1f}kWh",
            "power_kw": "{:.0f}kW",
            "user_score": "{:.1f}",
        }),
        use_container_width=True,
        height=400,
    )


# =============================================================================
# 模块 3: 品牌对比分析
# =============================================================================


def render_brand_comparison(df: pd.DataFrame):
    """渲染品牌对比分析"""
    st.header("📊 品牌对比分析")
    st.markdown("---")

    # 品牌选择
    all_brands = sorted(df["brand"].unique().tolist())
    default_brands = ["比亚迪", "特斯拉", "蔚来", "小鹏", "理想"]
    default_brands = [b for b in default_brands if b in all_brands]

    selected_brands = st.multiselect(
        "选择对比品牌 (2-5个)",
        options=all_brands,
        default=default_brands[:5] if default_brands else all_brands[:3],
        max_selections=5,
    )

    if len(selected_brands) < 2:
        st.warning("请至少选择 2 个品牌进行对比")
        return

    compare_df = df[df["brand"].isin(selected_brands)]

    st.markdown("---")

    # 第一行：雷达图 + 关键指标对比
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🎯 多维度综合雷达图")

        # 计算每个品牌在各维度的均值
        radar_dims = {
            "续航里程 (km)": "range_km",
            "电池容量 (kWh)": "battery_capacity",
            "动力功率 (kW)": "power_kw",
            "车身空间 (mm)": "length_mm",
            "智能驾驶": "adas_level_num",
        }
        radar_dims = {k: v for k, v in radar_dims.items() if v in compare_df.columns}

        if radar_dims:
            # 对各维度做 min-max 归一化便于雷达图展示
            radar_data = []
            for brand in selected_brands:
                brand_df = compare_df[compare_df["brand"] == brand]
                values = []
                for dim_name, dim_col in radar_dims.items():
                    raw = brand_df[dim_col].mean()
                    # 归一化到 0-100
                    global_min = compare_df[dim_col].min()
                    global_max = compare_df[dim_col].max()
                    if global_max > global_min:
                        normalized = (raw - global_min) / (global_max - global_min) * 100
                    else:
                        normalized = 50
                    values.append(normalized)
                radar_data.append({"brand": brand, "values": values})

            fig = go.Figure()
            for rd in radar_data:
                fig.add_trace(go.Scatterpolar(
                    r=rd["values"],
                    theta=list(radar_dims.keys()),
                    fill="toself",
                    name=rd["brand"],
                    opacity=0.6,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
                height=450,
                margin=dict(l=40, r=40, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("缺少雷达图所需的维度数据")

    with col_right:
        st.subheader("📊 关键指标对比")

        # 选择指标
        metric_col = st.selectbox(
            "选择对比指标",
            options=["price_median", "range_km", "battery_capacity",
                     "power_kw", "user_score", "accel_0_100"],
            format_func=lambda x: {
                "price_median": "价格 (万元)",
                "range_km": "续航里程 (km)",
                "battery_capacity": "电池容量 (kWh)",
                "power_kw": "电机功率 (kW)",
                "user_score": "用户评分",
                "accel_0_100": "百公里加速 (s)",
            }.get(x, x),
        )

        if metric_col in compare_df.columns:
            fig = px.box(
                compare_df,
                x="brand",
                y=metric_col,
                color="brand",
                points="outliers",
                category_orders={"brand": selected_brands},
            )
            fig.update_layout(
                height=450,
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    # 第二行：价格分布对比
    st.markdown("---")
    st.subheader("💰 价格区间分布对比")

    # 分组柱状图：各品牌在各价格区间的车型数量
    if "price_category" in compare_df.columns:
        cat_col = "price_category"
    else:
        bins = [0, 10, 20, 35, 60, float("inf")]
        labels = ["<10万", "10-20万", "20-35万", "35-60万", ">60万"]
        compare_df = compare_df.copy()
        compare_df["price_cat_temp"] = pd.cut(
            compare_df["price_median"], bins=bins, labels=labels
        )
        cat_col = "price_cat_temp"

    cross_tab = pd.crosstab(compare_df["brand"], compare_df[cat_col])
    cross_tab = cross_tab.reindex(selected_brands)

    fig = px.bar(
        cross_tab,
        barmode="group",
        labels={"value": "车型数", "brand": "品牌", cat_col: "价格区间"},
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 模块 4: 价格预测器
# =============================================================================


def render_price_predictor(df: pd.DataFrame):
    """渲染价格预测器"""
    st.header("💰 新能源汽车价格预测器")
    st.markdown("---")

    models = load_models()

    if not models["loaded"]:
        st.warning("""
        ⚠️ **预测模型未加载**

        请先运行以下命令训练并保存模型：
        ```
        cd EV_Market_Analysis
        python src/generate_sample_data.py
        python src/model.py
        ```
        模型文件将自动保存到 `app/` 目录下。
        """)
        return

    st.markdown("""
    基于 **随机森林回归模型 (R² = 0.912)**，输入车型关键参数即可实时预测其市场价格。
    """)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📝 输入车型参数")

        # 输入参数
        col_a, col_b = st.columns(2)

        with col_a:
            battery_capacity = st.slider(
                "电池容量 (kWh)",
                min_value=10.0, max_value=150.0,
                value=60.0, step=0.5,
                help="动力电池组的额定容量"
            )
            power_kw = st.slider(
                "电机功率 (kW)",
                min_value=20, max_value=500,
                value=150, step=5,
            )
            range_km = st.slider(
                "续航里程 (km)",
                min_value=100, max_value=1000,
                value=500, step=10,
                help="NEDC/CLTC 工况下的综合续航里程"
            )
            length_mm = st.slider(
                "车身长度 (mm)",
                min_value=2800, max_value=5500,
                value=4700, step=50,
            )

        with col_b:
            torque_nm = st.slider(
                "扭矩 (Nm)",
                min_value=50, max_value=1000,
                value=250, step=10,
            )
            accel = st.slider(
                "百公里加速 (s)",
                min_value=1.5, max_value=14.0,
                value=7.5, step=0.1,
            )
            wheelbase_mm = st.slider(
                "轴距 (mm)",
                min_value=1800, max_value=3500,
                value=2800, step=50,
            )
            width_mm = st.slider(
                "车身宽度 (mm)",
                min_value=1500, max_value=2100,
                value=1850, step=10,
            )

        height_mm = st.slider(
            "车身高度 (mm)",
            min_value=1200, max_value=2000,
            value=1500, step=10,
        )

        adas_options = {"L0": 0, "L1": 1, "L2": 2, "L2+": 2.5, "L3": 3}
        adas_display = st.selectbox(
            "智能驾驶等级",
            options=list(adas_options.keys()),
            index=2,
        )
        adas_level = adas_options[adas_display]

        seats = st.slider("座位数", 2, 7, 5)

        # 品牌选择
        all_brands = sorted(df["brand"].unique().tolist())
        selected_brand = st.selectbox("品牌", options=all_brands, index=0)

        # 预测按钮
        predict_btn = st.button("🔮 预测价格", type="primary", use_container_width=True)

    with col_right:
        st.subheader("📊 预测结果")

        if predict_btn:
            with st.spinner("正在预测..."):
                try:
                    # 构建特征向量
                    rf_model = models["rf_model"]
                    scaler = models["scaler"]
                    features = models["features"]

                    # 查找品牌的编码
                    brand_encoded = hash(selected_brand) % 78  # 简化的编码

                    # 构建输入
                    input_dict = {
                        "battery_capacity": battery_capacity,
                        "range_km": range_km,
                        "power_kw": power_kw,
                        "torque_nm": torque_nm,
                        "accel_0_100": accel,
                        "length_mm": length_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                        "wheelbase_mm": wheelbase_mm,
                        "adas_level_num": adas_level,
                        "seats": seats,
                        "range_price_ratio": range_km / max(battery_capacity * 6, 0.01),
                        "brand_premium_index": 0.0,
                        "power_score": 0.0,
                        "space_index": 0.0,
                        "tech_score": 0.0,
                        "brand_encoded": brand_encoded,
                        "battery_type_encoded": 1,
                        "body_type_encoded": 2,
                        "energy_type_encoded": 0,
                    }

                    # 只取模型需要的特征
                    input_vector = []
                    for f in features:
                        if f in input_dict:
                            input_vector.append(input_dict[f])
                        else:
                            input_vector.append(0.0)

                    X_input = np.array([input_vector])

                    # 标准化 + 预测
                    X_scaled = scaler.transform(X_input)
                    predicted_price = rf_model.predict(X_scaled)[0]

                    # ---- 显示结果 ----
                    st.metric(
                        label="预测价格",
                        value=f"{predicted_price:.1f} 万元",
                        delta=None,
                    )

                    # 置信区间（基于树预测的分布）
                    tree_preds = np.array([
                        tree.predict(X_scaled)[0]
                        for tree in rf_model.estimators_
                    ])
                    lower = np.percentile(tree_preds, 5)
                    upper = np.percentile(tree_preds, 95)

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #10b981, #059669);
                                border-radius: 12px; padding: 16px; color: white; margin-top: 8px;">
                        <strong>90% 置信区间:</strong><br>
                        <span style="font-size: 1.2rem;">
                            {lower:.1f} 万 ~ {upper:.1f} 万
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    # 与市面同类车对比
                    brand_avg = df[df["brand"] == selected_brand]["price_median"].mean()
                    market_avg = df["price_median"].mean()

                    st.markdown("---")
                    st.subheader("📈 价格对比")

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("预测价格", f"{predicted_price:.1f}万")
                    col_b.metric(f"{selected_brand}品牌均价",
                                 f"{brand_avg:.1f}万",
                                 delta=f"{(predicted_price - brand_avg):+.1f}万")
                    col_c.metric("市场均价",
                                 f"{market_avg:.1f}万",
                                 delta=f"{(predicted_price - market_avg):+.1f}万")

                except Exception as e:
                    st.error(f"预测失败: {str(e)}")
                    st.info("请确认模型文件 (rf_price_model.pkl, scaler.pkl) 位于 app/ 目录下")
        else:
            # 未点击预测时，展示特征重要性
            st.info("👆 请在左侧输入车型参数后点击「预测价格」按钮")

            # 展示特征重要性图表
            if models.get("metadata") and models["metadata"].get("feature_importance"):
                fi_data = models["metadata"]["feature_importance"]
                if fi_data:
                    fi_df = pd.DataFrame(fi_data).head(10)

                    fig = px.bar(
                        fi_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        color="importance",
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(
                        title="特征重要性 (Top 10)",
                        height=350,
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # 底部：聚类分析可视化
    st.markdown("---")
    st.subheader("🔬 车型聚类分析 (K-Means, K=3)")

    if "cluster_name" in df.columns and "pca_x" in df.columns:
        fig = px.scatter(
            df.sample(min(1500, len(df)), random_state=42),
            x="pca_x",
            y="pca_y",
            color="cluster_name",
            hover_data=["brand", "model", "price_median", "range_km"],
            labels={
                "pca_x": "第一主成分 (经济性 → 性能)",
                "pca_y": "第二主成分 (尺寸/续航)",
                "cluster_name": "细分市场",
            },
            color_discrete_map={
                "经济代步型": "#22c55e",
                "中端家用型": "#3b82f6",
                "高端性能型": "#ef4444",
            },
        )
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("聚类数据未包含在数据集中。运行 `python src/model.py` 训练模型后将包含聚类标签。")


# =============================================================================
# 模块 5: 区域市场分析
# =============================================================================


def render_regional_analysis(df: pd.DataFrame):
    """渲染区域市场分析"""
    st.header("🗺️ 区域市场分析")
    st.markdown("---")

    regional_df = load_regional_data()

    if regional_df.empty:
        st.warning("""
        ⚠️ 区域经济数据未加载。

        请先运行:
        ```
        python src/generate_sample_data.py
        ```
        """)
        return

    # ---- 地理热力图 ----
    st.subheader("🇨🇳 全国新能源汽车消费热力图")

    map_metric = st.selectbox(
        "选择地图指标",
        options=["ev_annual_sales_10k", "charger_count_per_10k",
                 "gdp_per_capita", "urban_income", "winter_avg_temp"],
        format_func=lambda x: {
            "ev_annual_sales_10k": "新能源车年销量 (万辆)",
            "charger_count_per_10k": "充电桩密度 (公共桩/万人)",
            "gdp_per_capita": "人均 GDP (万元)",
            "urban_income": "城镇居民可支配收入 (万元)",
            "winter_avg_temp": "冬季平均温度 (°C)",
        }.get(x, x),
    )

    # 使用 Plotly Choropleth
    metric_label_map = {
        "ev_annual_sales_10k": "新能源车年销量 (万辆)",
        "charger_count_per_10k": "充电桩密度 (公共桩/万人)",
        "gdp_per_capita": "人均 GDP (万元)",
        "urban_income": "城镇居民可支配收入 (万元)",
        "winter_avg_temp": "冬季平均温度 (°C)",
    }

    fig = px.choropleth(
        regional_df,
        locations="province",
        locationmode="country names",
        color=map_metric,
        hover_name="province",
        hover_data={
            "ev_annual_sales_10k": ":.1f",
            "charger_count_per_10k": ":.1f",
            "urban_income": ":.1f",
            "winter_avg_temp": ":.1f",
        },
        color_continuous_scale="RdYlGn" if map_metric != "winter_avg_temp" else "RdBu_r",
        labels=metric_label_map,
        # 中国地图投影
        scope="asia",
    )
    fig.update_geos(
        visible=False,
        lonaxis_range=[73, 135],
        lataxis_range=[15, 55],
        showcountries=True,
        countrycolor="rgba(0,0,0,0.2)",
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(
            center={"lat": 35, "lon": 105},
            projection_scale=4,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- 区域指标对比 ----
    st.markdown("---")
    st.subheader("📊 区域关键指标对比")

    # Top 10 省份
    top_provinces = regional_df.nlargest(10, "ev_annual_sales_10k")["province"].tolist()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        fig = px.bar(
            regional_df.nlargest(15, "ev_annual_sales_10k"),
            x="province",
            y="ev_annual_sales_10k",
            color="ev_annual_sales_10k",
            color_continuous_scale="Blues",
            labels={
                "province": "省市",
                "ev_annual_sales_10k": "新能源车年销量 (万辆)",
            },
        )
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.scatter(
            regional_df,
            x="charger_count_per_10k",
            y="ev_annual_sales_10k",
            size="gdp_trillion",
            color="winter_avg_temp",
            hover_name="province",
            labels={
                "charger_count_per_10k": "充电桩密度 (公共桩/万人)",
                "ev_annual_sales_10k": "新能源车年销量 (万辆)",
                "gdp_trillion": "GDP (万亿)",
                "winter_avg_temp": "冬季均温 (°C)",
            },
            color_continuous_scale="RdBu_r",
        )
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ---- 政策影响分析 ----
    st.markdown("---")
    st.subheader("📋 区域数据详情")

    st.dataframe(
        regional_df.sort_values("ev_annual_sales_10k", ascending=False),
        use_container_width=True,
        height=400,
        column_config={
            "province": "省市",
            "gdp_trillion": st.column_config.NumberColumn("GDP(万亿)", format="%.2f"),
            "gdp_per_capita": st.column_config.NumberColumn("人均GDP(万)", format="%.1f"),
            "urban_income": st.column_config.NumberColumn("人均收入(万)", format="%.1f"),
            "charger_count_per_10k": st.column_config.NumberColumn("充电桩/万人", format="%.1f"),
            "ev_annual_sales_10k": st.column_config.NumberColumn("年销量(万辆)", format="%.1f"),
            "winter_avg_temp": st.column_config.NumberColumn("冬季均温(°C)", format="%.1f"),
            "is_plate_restricted": st.column_config.CheckboxColumn("限牌城市"),
        },
    )


# =============================================================================
# 主程序
# =============================================================================


def main():
    """EV-Insight 主入口"""

    # ---- 侧边栏 ----
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:10px 0;">
            <span style="font-size:3rem;">🚗</span>
            <h2 style="margin:0;">EV-Insight</h2>
            <p style="color:#9ca3af;font-size:0.85rem;">
                新能源汽车市场洞察平台
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # 导航菜单
        page = st.radio(
            "📌 导航菜单",
            options=[
                "🏠 市场总览仪表盘",
                "🔍 车型探索器",
                "📊 品牌对比分析",
                "💰 价格预测器",
                "🗺️ 区域市场分析",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # 全局筛选器
        st.markdown("### ⚙️ 全局筛选")

        # 加载数据
        df = load_data()
        if df.empty:
            st.error("请先生成数据文件")
            st.stop()

        # 品牌多选
        all_brands_global = sorted(df["brand"].unique().tolist())
        if "global_brands" not in st.session_state:
            st.session_state.global_brands = all_brands_global

        st.session_state.global_brands = st.multiselect(
            "品牌筛选",
            options=all_brands_global,
            default=st.session_state.global_brands[:10] if len(st.session_state.global_brands) > 10 else st.session_state.global_brands,
        )

        # 价格范围
        price_range_global = st.slider(
            "价格范围 (万元)",
            min_value=0.0,
            max_value=float(df["price_median"].max()),
            value=(0.0, float(df["price_median"].max())),
            step=5.0,
        )

        # 续航范围
        range_global = st.slider(
            "续航范围 (km)",
            min_value=0,
            max_value=int(df["range_km"].max()),
            value=(0, int(df["range_km"].max())),
            step=50,
        )

        st.markdown("---")
        st.markdown("""
        <div class="footer">
            <p>📊 基于 Plotly + Streamlit 构建</p>
            <p>📚 数据可视化课程大作业</p>
            <p>© 2025 EV-Insight</p>
        </div>
        """, unsafe_allow_html=True)

    # ---- 全局数据筛选 ----
    df_filtered = df.copy()
    if st.session_state.global_brands:
        df_filtered = df_filtered[df_filtered["brand"].isin(st.session_state.global_brands)]
    df_filtered = df_filtered[
        (df_filtered["price_median"] >= price_range_global[0])
        & (df_filtered["price_median"] <= price_range_global[1])
        & (df_filtered["range_km"] >= range_global[0])
        & (df_filtered["range_km"] <= range_global[1])
    ]

    # ---- 主内容区 ----
    st.markdown('<p class="main-title">🚗 EV-Insight</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">新能源汽车市场洞察平台 | 基于多源数据融合的市场分析与价格预测</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # 根据导航渲染对应模块
    if "市场总览" in page:
        render_overview(df_filtered)
    elif "车型探索" in page:
        render_vehicle_explorer(df_filtered)
    elif "品牌对比" in page:
        render_brand_comparison(df_filtered)
    elif "价格预测" in page:
        render_price_predictor(df_filtered)
    elif "区域市场" in page:
        render_regional_analysis(df_filtered)


if __name__ == "__main__":
    main()
