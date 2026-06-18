"""
=============================================================================
EV-Insight 机器学习建模模块
=============================================================================
实现论文 3.2 节描述的数据建模方法：

1. K-Means 聚类 - 车型细分市场识别（3 类）
2. 随机森林回归 - 新能源汽车价格预测模型
3. 梯度提升回归 - 对比模型
4. 线性回归 - 基线模型
5. 特征重要性分析

使用方法：
    python model.py
    → 训练所有模型
    → 保存 model (.pkl) 和 scaler (.pkl) 到 app/ 目录
=============================================================================
"""

import os
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, r2_score,
                              mean_absolute_percentage_error,
                              silhouette_score)

warnings.filterwarnings("ignore")

# =============================================================================
# 配置
# =============================================================================

# 特征列定义
BASE_FEATURES = [
    "battery_capacity",    # 电池容量 (kWh)
    "range_km",            # 续航里程 (km)
    "power_kw",            # 电机功率 (kW)
    "torque_nm",           # 扭矩 (Nm)
    "accel_0_100",         # 百公里加速 (s)
    "length_mm",           # 车长 (mm)
    "width_mm",            # 车宽 (mm)
    "height_mm",           # 车高 (mm)
    "wheelbase_mm",        # 轴距 (mm)
    "adas_level_num",      # 智能驾驶等级 (数值化)
    "seats",               # 座位数
]

DERIVED_FEATURES = [
    "range_price_ratio",   # 续航价格比
    "brand_premium_index", # 品牌溢价指数
    "power_score",         # 动力性能综合分
    "space_index",         # 空间舒适度指数
    "tech_score",          # 技术先进度评分
]

ENCODED_FEATURES = [
    "brand_encoded",
    "battery_type_encoded",
    "body_type_encoded",
    "energy_type_encoded",
]

# 聚类特征（不含品牌溢价等衍生特征，避免循环依赖）
CLUSTER_FEATURES = [
    "price_median", "range_km", "battery_capacity", "power_kw",
    "length_mm", "adas_level_num", "accel_0_100",
]

# =============================================================================
# 辅助函数
# =============================================================================


def get_all_features(df: pd.DataFrame) -> list:
    """获取 DataFrame 中实际存在的所有特征列"""
    all_candidates = BASE_FEATURES + DERIVED_FEATURES + ENCODED_FEATURES
    return [f for f in all_candidates if f in df.columns]


def prepare_features(df: pd.DataFrame, features: list,
                     target: str = "price_median") -> tuple:
    """
    准备模型输入特征和标签。

    Parameters
    ----------
    df : pd.DataFrame
    features : list
        特征列名列表
    target : str
        目标变量列名

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, y)
    """
    # 只保留存在的列
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()

    # 填充残留的 NaN
    X = X.fillna(X.median())

    y = df[target].copy()
    y = y.fillna(y.median())

    return X.values, y.values


# =============================================================================
# 1. K-Means 车型聚类
# =============================================================================


def train_kmeans_clustering(df: pd.DataFrame,
                            n_clusters: int = 3,
                            random_state: int = 42) -> dict:
    """
    训练 K-Means 聚类模型，识别车型细分市场。

    根据论文 3.2.1 节：
    - K=3，轮廓系数约 0.457
    - 聚类 0: 经济代步型 (占比 ~38%)
    - 聚类 1: 中端家用型 (占比 ~45%)
    - 聚类 2: 高端性能型 (占比 ~17%)

    Parameters
    ----------
    df : pd.DataFrame
        预处理后的数据
    n_clusters : int
        聚类数
    random_state : int

    Returns
    -------
    dict
        {
            "model": KMeans 模型,
            "scaler": StandardScaler,
            "labels": 聚类标签,
            "silhouette": 轮廓系数,
            "cluster_stats": 各类别统计信息,
            "pca": PCA 模型 (用于 2D 可视化),
            "X_pca": PCA 降维后的坐标,
        }
    """
    print("\n" + "=" * 60)
    print("🔵 1. K-Means 车型聚类分析")
    print("=" * 60)

    # 准备数据
    available_features = [f for f in CLUSTER_FEATURES if f in df.columns]
    print(f"   使用特征 ({len(available_features)}): {available_features}")

    X = df[available_features].copy()
    X = X.fillna(X.median())

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练 K-Means
    print(f"   训练 K-Means (K={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,
                    n_init=20, max_iter=500)
    labels = kmeans.fit_predict(X_scaled)

    # 轮廓系数
    sil_score = silhouette_score(X_scaled, labels)
    print(f"   轮廓系数 (Silhouette Score): {sil_score:.4f}")

    # 各类别统计
    cluster_stats = []
    cluster_names = {
        0: "经济代步型",
        1: "中端家用型",
        2: "高端性能型",
    }

    df_with_labels = df.copy()
    df_with_labels["cluster"] = labels

    for c in range(n_clusters):
        cluster_data = df_with_labels[df_with_labels["cluster"] == c]
        stats = {
            "cluster_id": c,
            "cluster_name": cluster_names.get(c, f"聚类{c}"),
            "count": len(cluster_data),
            "percentage": len(cluster_data) / len(df) * 100,
            "avg_price": cluster_data["price_median"].mean(),
            "avg_range": cluster_data["range_km"].mean(),
            "avg_power": cluster_data["power_kw"].mean(),
            "avg_length": cluster_data["length_mm"].mean(),
            "avg_adas": cluster_data["adas_level_num"].mean(),
            "avg_battery": cluster_data["battery_capacity"].mean(),
        }
        cluster_stats.append(stats)
        print(f"\n   聚类 {c} - {stats['cluster_name']}:")
        print(f"     车型数:       {stats['count']} ({stats['percentage']:.1f}%)")
        print(f"     平均价格:     {stats['avg_price']:.1f} 万")
        print(f"     平均续航:     {stats['avg_range']:.0f} km")
        print(f"     平均车身长度: {stats['avg_length']:.0f} mm")
        print(f"     平均 ADAS:    {stats['avg_adas']:.1f}")

    # PCA 降维用于可视化
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n   PCA 解释方差比: {pca.explained_variance_ratio_}")

    # 分配聚类名称（根据均价自动映射）
    avg_prices = [s["avg_price"] for s in cluster_stats]
    sorted_indices = np.argsort(avg_prices)
    name_map = {
        sorted_indices[0]: "经济代步型",
        sorted_indices[1]: "中端家用型",
        sorted_indices[2]: "高端性能型",
    }
    for s in cluster_stats:
        s["cluster_name"] = name_map.get(s["cluster_id"], s["cluster_name"])

    return {
        "model": kmeans,
        "scaler": scaler,
        "labels": labels,
        "silhouette": sil_score,
        "cluster_stats": cluster_stats,
        "pca": pca,
        "X_pca": X_pca,
    }


# =============================================================================
# 2. 价格预测模型训练
# =============================================================================


def train_price_prediction_models(df: pd.DataFrame,
                                  test_size: float = 0.2,
                                  random_state: int = 42) -> dict:
    """
    训练并对比三种价格预测模型。

    根据论文 3.2.2 节：
    - 线性回归 (基线):   R² ≈ 0.783, RMSE ≈ 6.82
    - 随机森林 (最优):   R² ≈ 0.912, RMSE ≈ 3.95
    - 梯度提升 (对比):   R² ≈ 0.897, RMSE ≈ 4.28

    Parameters
    ----------
    df : pd.DataFrame
    test_size : float
    random_state : int

    Returns
    -------
    dict
        包含所有模型、scaler、评估指标和特征重要性
    """
    print("\n" + "=" * 60)
    print("🟢 2. 价格预测模型训练")
    print("=" * 60)

    # 准备特征
    features = get_all_features(df)
    print(f"   使用特征 ({len(features)}):")
    for f in features:
        print(f"     - {f}")

    X, y = prepare_features(df, features, target="price_median")

    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\n   训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- 2.1 线性回归 (基线) ----
    print("\n   [1/3] 训练线性回归 (基线模型)...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)
    lr_mape = mean_absolute_percentage_error(y_test, lr_pred) * 100
    print(f"     RMSE: {lr_rmse:.2f} 万, R²: {lr_r2:.3f}, MAPE: {lr_mape:.1f}%")

    # ---- 2.2 随机森林 (核心模型) ----
    print("\n   [2/3] 训练随机森林回归...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mape = mean_absolute_percentage_error(y_test, rf_pred) * 100
    print(f"     RMSE: {rf_rmse:.2f} 万, R²: {rf_r2:.3f}, MAPE: {rf_mape:.1f}%")

    # ---- 2.3 梯度提升 ----
    print("\n   [3/3] 训练梯度提升回归...")
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=random_state,
    )
    gb.fit(X_train_scaled, y_train)
    gb_pred = gb.predict(X_test_scaled)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    gb_r2 = r2_score(y_test, gb_pred)
    gb_mape = mean_absolute_percentage_error(y_test, gb_pred) * 100
    print(f"     RMSE: {gb_rmse:.2f} 万, R²: {gb_r2:.3f}, MAPE: {gb_mape:.1f}%")

    # ---- 模型对比表 ----
    print("\n   " + "-" * 55)
    print(f"   {'模型':<20} {'RMSE(万)':<12} {'R²':<10} {'MAPE(%)':<10}")
    print("   " + "-" * 55)
    print(f"   {'线性回归':<20} {lr_rmse:<12.2f} {lr_r2:<10.3f} {lr_mape:<10.1f}")
    print(f"   {'随机森林 ★':<20} {rf_rmse:<12.2f} {rf_r2:<10.3f} {rf_mape:<10.1f}")
    print(f"   {'梯度提升':<20} {gb_rmse:<12.2f} {gb_r2:<10.3f} {gb_mape:<10.1f}")
    print("   " + "-" * 55)

    # ---- 特征重要性 ----
    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n   📊 随机森林特征重要性 (Top 10):")
    for i, row in feature_importance.head(10).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"     {row['feature']:<25} {row['importance']:.4f}  {bar}")

    return {
        # 模型
        "lr_model": lr,
        "rf_model": rf,
        "gb_model": gb,
        # 预处理器
        "scaler": scaler,
        "features": features,
        # 评估指标
        "metrics": {
            "linear_regression": {"rmse": lr_rmse, "r2": lr_r2, "mape": lr_mape},
            "random_forest": {"rmse": rf_rmse, "r2": rf_r2, "mape": rf_mape},
            "gradient_boosting": {"rmse": gb_rmse, "r2": gb_r2, "mape": gb_mape},
        },
        # 特征重要性
        "feature_importance": feature_importance,
    }


# =============================================================================
# 3. 保存模型
# =============================================================================


def save_models(results: dict, output_dir: str = None):
    """
    保存训练好的模型和预处理器到磁盘。

    保存内容：
    - rf_price_model.pkl:    随机森林回归模型
    - scaler.pkl:            StandardScaler
    - kmeans_model.pkl:      K-Means 聚类模型
    - cluster_scaler.pkl:    聚类用 StandardScaler
    - feature_names.pkl:     特征名列表
    - model_metadata.pkl:    模型元数据（评估指标等）
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "app"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_saved = []

    # 随机森林模型（主力）
    if "rf_model" in results:
        path = output_dir / "rf_price_model.pkl"
        joblib.dump(results["rf_model"], path)
        files_saved.append(path)

    # 梯度提升模型（备选）
    if "gb_model" in results:
        path = output_dir / "gb_price_model.pkl"
        joblib.dump(results["gb_model"], path)
        files_saved.append(path)

    # 价格预测 Scaler
    if "price_scaler" in results:
        path = output_dir / "scaler.pkl"
        joblib.dump(results["price_scaler"], path)
        files_saved.append(path)
    elif "scaler" in results:
        path = output_dir / "scaler.pkl"
        joblib.dump(results["scaler"], path)
        files_saved.append(path)

    # K-Means 模型
    if "kmeans" in results:
        kmeans_result = results["kmeans"]
        path = output_dir / "kmeans_model.pkl"
        joblib.dump(kmeans_result["model"], path)
        files_saved.append(path)

        path = output_dir / "cluster_scaler.pkl"
        joblib.dump(kmeans_result["scaler"], path)
        files_saved.append(path)

    # 特征名列表
    if "features" in results:
        path = output_dir / "feature_names.pkl"
        joblib.dump(results["features"], path)
        files_saved.append(path)

    # 元数据
    metadata = {
        "features": results.get("features", []),
        "metrics": results.get("metrics", {}),
        "feature_importance": None,
    }
    if "feature_importance" in results:
        metadata["feature_importance"] = results["feature_importance"].to_dict("records")
    if "cluster_stats" in results.get("kmeans", {}):
        metadata["cluster_stats"] = results["kmeans"]["cluster_stats"]

    path = output_dir / "model_metadata.pkl"
    joblib.dump(metadata, path)
    files_saved.append(path)

    print(f"\n💾 模型已保存到: {output_dir}")
    for f in files_saved:
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name} ({size_kb:.0f} KB)")

    return files_saved


# =============================================================================
# 主程序
# =============================================================================


def main():
    """完整模型训练流程"""
    # 确定路径
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "ev_data_cleaned.csv"

    print("=" * 60)
    print("🚗 EV-Insight 模型训练流水线")
    print("=" * 60)

    # 加载数据
    if not data_path.exists():
        print(f"\n❌ 数据文件不存在: {data_path}")
        print("   请先运行:  python src/generate_sample_data.py")
        sys.exit(1)

    print(f"\n📂 加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   记录数: {len(df)}, 字段数: {len(df.columns)}")

    # ---- 1. K-Means 聚类 ----
    kmeans_result = train_kmeans_clustering(df, n_clusters=3)

    # 为后续分析添加聚类标签
    df["cluster"] = kmeans_result["labels"]
    df["cluster_name"] = df["cluster"].map({
        s["cluster_id"]: s["cluster_name"]
        for s in kmeans_result["cluster_stats"]
    })
    df["pca_x"] = kmeans_result["X_pca"][:, 0]
    df["pca_y"] = kmeans_result["X_pca"][:, 1]

    # ---- 2. 价格预测模型 ----
    price_result = train_price_prediction_models(df)

    # ---- 3. 保存模型 ----
    results = {
        "kmeans": kmeans_result,
        **price_result,
    }
    save_models(results)

    # ---- 4. 最终输出 ----
    print("\n" + "=" * 60)
    print("✅ 模型训练完成！")
    print("=" * 60)
    print(f"\n📊 模型性能总结:")
    print(f"   K-Means 轮廓系数:  {kmeans_result['silhouette']:.4f}")
    metrics = price_result["metrics"]
    print(f"   随机森林 R²:       {metrics['random_forest']['r2']:.3f}")
    print(f"   随机森林 RMSE:     {metrics['random_forest']['rmse']:.2f} 万")
    print(f"\n🚀 下一步:")
    print(f"   streamlit run app/app.py")


if __name__ == "__main__":
    main()
