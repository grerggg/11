"""
=============================================================================
EV-Insight 数据预处理与特征工程模块
=============================================================================
实现论文 2.3 节描述的数据清洗、标准化、缺失值处理和特征工程。

核心功能：
1. 数据清洗：文本字段清理、数值字段标准化、异常值检测与处理
2. 缺失值处理：基于品牌中位数的分层填充策略
3. 特征工程：续航价格比、品牌溢价指数、动力性能综合分等衍生特征
4. 编码转换：Label Encoding / One-Hot Encoding / 智能驾驶等级量化
=============================================================================
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# =============================================================================
# 1. 文本与数值解析
# =============================================================================


def parse_price_column(price_series: pd.Series) -> pd.DataFrame:
    """
    解析价格字符串，如 "14.98-19.98万" → 提取 low/high/median 三个数值列。

    Parameters
    ----------
    price_series : pd.Series
        包含价格字符串的 Series

    Returns
    -------
    pd.DataFrame
        price_low, price_high, price_median 三列
    """
    result = pd.DataFrame(index=price_series.index)

    result["price_low"] = np.nan
    result["price_high"] = np.nan

    # 模式1: "14.98-19.98万"
    pattern_range = r"([\d.]+)\s*(?:-|~)\s*([\d.]+)\s*万"
    # 模式2: "14.98万"
    pattern_single = r"^([\d.]+)\s*万"

    for idx, val in price_series.items():
        if pd.isna(val) or not isinstance(val, str):
            continue
        match = re.search(pattern_range, str(val))
        if match:
            result.loc[idx, "price_low"] = float(match.group(1))
            result.loc[idx, "price_high"] = float(match.group(2))
        else:
            match = re.search(pattern_single, str(val))
            if match:
                p = float(match.group(1))
                result.loc[idx, "price_low"] = p
                result.loc[idx, "price_high"] = p

    # 中位数价格
    result["price_median"] = (result["price_low"] + result["price_high"]) / 2

    return result


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    清洗数值列：去除单位、提取数值。

    支持的格式: "123 km", "500km", "50 kWh", "150kW", "4.5s", "4500mm" 等
    """
    if series.dtype in [np.float64, np.int64, np.float32, np.int32]:
        return series

    def extract(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        # 提取第一个数字
        match = re.search(r"([\d.]+)", str(val))
        return float(match.group(1)) if match else np.nan

    return series.apply(extract)


def standardize_brand_name(name: str) -> str:
    """
    统一品牌名称。

    处理常见的变体写法：
    - "BYD" / "比亜迪" → "比亚迪"
    - "Tesla" → "特斯拉"
    - "NIO" → "蔚来"
    - 去除多余空格和特殊字符
    """
    if pd.isna(name):
        return "未知"

    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)

    # 品牌名称映射表
    brand_map = {
        "byd": "比亚迪",
        "比亜迪": "比亚迪",
        "tesla": "特斯拉",
        "nio": "蔚来",
        "xpeng": "小鹏",
        "li auto": "理想",
        "lixiang": "理想",
        "wuling": "五菱",
        "aion": "广汽埃安",
        "zeekr": "极氪",
        "voyah": "岚图",
        "avatr": "阿维塔",
        "deepal": "深蓝",
        "leapmotor": "零跑",
        "neta": "哪吒",
        "xiaomi": "小米汽车",
        "denza": "腾势",
        "yangwang": "仰望",
        "volkswagen": "大众ID",
        "vw": "大众ID",
        "toyota": "丰田",
        "honda": "本田",
        "nissan": "日产",
        "bmw": "宝马",
        "mercedes": "奔驰",
        "benz": "奔驰",
        "audi": "奥迪",
        "porsche": "保时捷",
        "volvo": "沃尔沃",
        "cadillac": "凯迪拉克",
        "lexus": "雷克萨斯",
    }
    return brand_map.get(name.lower(), name)


# =============================================================================
# 2. 缺失值处理
# =============================================================================


def fill_missing_by_brand(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    按品牌中位数填充缺失值。

    策略：
    1. 先用同品牌中位数填充
    2. 若品牌级别也为空（整品牌缺失），用全局中位数填充

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        需要填充的列名

    Returns
    -------
    pd.Series
    """
    # 按品牌分组的中位数
    brand_medians = df.groupby("brand")[target_col].transform("median")
    global_median = df[target_col].median()

    filled = df[target_col].fillna(brand_medians)
    filled = filled.fillna(global_median)

    return filled


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    使用 IQR（四分位距）方法检测异常值。

    Parameters
    ----------
    series : pd.Series
    multiplier : float
        IQR 倍数，默认 1.5

    Returns
    -------
    pd.Series (bool)
        True 表示是异常值
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)


# =============================================================================
# 3. 特征工程
# =============================================================================


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建衍生特征。

    构建的特征（论文 2.3.3 节）：
    - range_price_ratio:      续航价格比 (km/万)
    - brand_premium_index:    品牌溢价指数
    - power_score:            动力性能综合分
    - space_index:            空间舒适度指数
    - tech_score:             技术先进度评分
    - price_category:         价格等级分类
    """
    df = df.copy()

    # ---- 3.1 续航价格比 ----
    df["range_price_ratio"] = np.where(
        df["price_median"] > 0,
        df["range_km"] / df["price_median"],
        np.nan,
    )

    # ---- 3.2 品牌溢价指数 ----
    # 计算每个品牌相对于整体市场中位数的偏差
    global_median_price = df["price_median"].median()
    brand_median_price = df.groupby("brand")["price_median"].transform("median")
    df["brand_premium_index"] = (brand_median_price - global_median_price) / global_median_price

    # ---- 3.3 动力性能综合分 ----
    # 对功率、扭矩、百公里加速标准化后加权合成
    power_z = (df["power_kw"] - df["power_kw"].mean()) / df["power_kw"].std()
    torque_z = (df["torque_nm"] - df["torque_nm"].mean()) / df["torque_nm"].std()
    # 加速时间：越小越好，取负值
    accel_z = -(df["accel_0_100"] - df["accel_0_100"].mean()) / df["accel_0_100"].std()
    df["power_score"] = (power_z.fillna(0) * 0.35
                         + torque_z.fillna(0) * 0.25
                         + accel_z.fillna(0) * 0.40)

    # ---- 3.4 空间舒适度指数 ----
    # 基于轴距和车身尺寸
    wb_z = (df["wheelbase_mm"] - df["wheelbase_mm"].mean()) / df["wheelbase_mm"].std()
    len_z = (df["length_mm"] - df["length_mm"].mean()) / df["length_mm"].std()
    wid_z = (df["width_mm"] - df["width_mm"].mean()) / df["width_mm"].std()
    df["space_index"] = (wb_z.fillna(0) * 0.5
                         + len_z.fillna(0) * 0.25
                         + wid_z.fillna(0) * 0.25)

    # ---- 3.5 技术先进度评分 ----
    # 综合电池能量密度、智能驾驶等级、快充能力等
    # 处理字符串型 adas_level (如 "L0", "L1", "L2")
    adas_map = {"L0": 0, "L0+": 0.5, "L1": 1, "L1+": 1.5,
                "L2": 2, "L2+": 2.5, "准L3": 2.8, "L3": 3}
    if df["adas_level"].dtype == object:
        adas_num = df["adas_level"].map(adas_map).fillna(1)
    else:
        adas_num = df["adas_level"].astype(float).fillna(1)
    adas_z = (adas_num - adas_num.mean()) / adas_num.std()

    # 电池能量密度近似: 容量/续航 * 常数
    energy_density_approx = df["battery_capacity"] / (df["range_km"] / 100)
    ed_z = (energy_density_approx - energy_density_approx.mean()) / energy_density_approx.std()

    df["tech_score"] = (adas_z.fillna(0) * 0.5
                        + ed_z.fillna(0) * 0.3
                        + (df["power_score"].fillna(0)) * 0.2)

    # ---- 3.6 价格等级分类 ----
    price_bins = [0, 10, 20, 35, 60, float("inf")]
    price_labels = ["经济型(<10万)", "入门型(10-20万)", "中端型(20-35万)",
                    "高端型(35-60万)", "豪华型(>60万)"]
    df["price_category"] = pd.cut(df["price_median"], bins=price_bins,
                                   labels=price_labels, right=True)

    return df


# =============================================================================
# 4. 编码转换
# =============================================================================


def encode_categorical(df: pd.DataFrame) -> tuple:
    """
    对类别变量进行编码。

    - brand → brand_encoded (Label Encoding)
    - battery_type → battery_type_encoded (Label Encoding)
    - body_type → body_type_encoded (Label Encoding)
    - energy_type → energy_type_encoded (Label Encoding)
    - adas_level → adas_level_num (数值化: L0=0, L1=1, L2=2, L2+=2.5, L3=3)

    Returns
    -------
    tuple[DataFrame, dict]
        (编码后的 DataFrame, 编码器字典)
    """
    df = df.copy()
    encoders = {}

    # Label Encoding
    label_cols = ["brand", "battery_type", "body_type", "energy_type", "price_category"]
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            # 处理 Categorical 类型：先转为 str
            if hasattr(df[col].dtype, 'categories'):
                df[col] = df[col].astype(str)
            # 先填充空值
            df[col] = df[col].fillna("未知")
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # 智能驾驶等级数值化
    adas_map = {
        "L0": 0, "L0+": 0.5,
        "L1": 1, "L1+": 1.5,
        "L2": 2, "L2+": 2.5,
        "准L3": 2.8, "L3": 3,
    }
    df["adas_level_num"] = df["adas_level"].map(adas_map).fillna(1.0)

    return df, encoders


# =============================================================================
# 5. 主预处理流程
# =============================================================================


def preprocess_ev_data(df: pd.DataFrame,
                       remove_outliers: bool = True) -> pd.DataFrame:
    """
    完整的新能源汽车数据预处理流程。

    严格按照论文 2.3 节的步骤执行：
    1. 文本字段清理 + 品牌名称统一
    2. 数值字段标准化（字符串 → 数值）
    3. 价格字段解析
    4. 缺失值处理（品牌中位数填充）
    5. 异常值检测与剔除
    6. 衍生特征构建
    7. 类别变量编码

    Parameters
    ----------
    df : pd.DataFrame
        原始数据
    remove_outliers : bool
        是否剔除异常值

    Returns
    -------
    pd.DataFrame
        清洗后的完整特征矩阵
    """
    print("=" * 60)
    print("🔧 EV-Insight 数据预处理流水线")
    print("=" * 60)

    df = df.copy()
    initial_count = len(df)
    print(f"\n📥 输入数据: {initial_count} 条记录")

    # ---- Step 1: 文本字段清理 ----
    print("  [1/7] 文本字段清理 & 品牌名称统一...")
    # 去除多余空格和 HTML 标签
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(r"<[^>]*>", "", regex=True)
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    if "brand" in df.columns:
        df["brand"] = df["brand"].apply(standardize_brand_name)

    # ---- Step 2: 数值字段标准化 ----
    print("  [2/7] 数值字段标准化...")
    numeric_cols_to_clean = [
        "range_km", "battery_capacity", "power_kw", "torque_nm",
        "accel_0_100", "length_mm", "width_mm", "height_mm",
        "wheelbase_mm", "user_score", "review_count", "seats",
    ]
    for col in numeric_cols_to_clean:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])

    # ---- Step 3: 价格字段解析 ----
    print("  [3/7] 价格字段解析...")
    if "guide_price_str" in df.columns:
        price_df = parse_price_column(df["guide_price_str"])
        df["price_low"] = price_df["price_low"]
        df["price_high"] = price_df["price_high"]
        df["price_median"] = price_df["price_median"]
    elif "guide_price" in df.columns:
        # 已经数值化的情况
        df["price_low"] = df["guide_price"]
        df["price_high"] = df["guide_price"]
        df["price_median"] = df["guide_price"]
    else:
        raise ValueError("数据中缺少价格字段 (guide_price_str 或 guide_price)")

    # ---- Step 4: 缺失值处理 ----
    print("  [4/7] 缺失值处理（品牌中位数分层填充）...")
    missing_report = {}
    fill_cols = ["price_median", "range_km", "battery_capacity",
                 "power_kw", "user_score"]

    for col in fill_cols:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_report[col] = missing_count
                df[col] = fill_missing_by_brand(df, col)

    if missing_report:
        for col, count in missing_report.items():
            print(f"     {col}: 填充 {count} 个缺失值 ({count/initial_count*100:.1f}%)")

    # ---- Step 5: 异常值检测与剔除 ----
    print("  [5/7] 异常值检测（IQR 方法）...")
    outlier_cols = ["price_median", "range_km", "power_kw", "battery_capacity"]
    outlier_mask = pd.Series(False, index=df.index)

    for col in outlier_cols:
        if col in df.columns:
            is_outlier = detect_outliers_iqr(df[col], multiplier=3.0)  # 用 3*IQR 减少误杀
            outlier_mask = outlier_mask | is_outlier

    outlier_count = outlier_mask.sum()
    if remove_outliers and outlier_count > 0:
        print(f"     检测到 {outlier_count} 条异常记录，正在剔除...")
        df = df[~outlier_mask].reset_index(drop=True)
        print(f"     剔除后剩余: {len(df)} 条记录")
    else:
        print(f"     检测到 {outlier_count} 条异常记录（保留用于分析）")

    # ---- Step 6: 衍生特征构建 ----
    print("  [6/7] 衍生特征构建...")
    df = create_derived_features(df)

    # ---- Step 7: 类别变量编码 ----
    print("  [7/7] 类别变量编码...")
    df, _ = encode_categorical(df)

    # ---- 最终汇总 ----
    final_count = len(df)
    print(f"\n✅ 预处理完成!")
    print(f"   原始记录: {initial_count}")
    print(f"   有效记录: {final_count}")
    print(f"   剔除记录: {initial_count - final_count}")
    print(f"   特征维度: {len(df.columns)}")
    print(f"   缺失值:   {df.isna().sum().sum()} (总计)")

    return df


# =============================================================================
# 6. 区域数据预处理
# =============================================================================


def preprocess_regional_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    区域经济数据预处理。

    Parameters
    ----------
    df : pd.DataFrame
        原始区域数据

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # 充电桩覆盖率指标
    if "charger_count_per_10k" in df.columns and "ev_annual_sales_10k" in df.columns:
        df["charger_coverage_ratio"] = np.where(
            df["ev_annual_sales_10k"] > 0,
            df["charger_count_per_10k"] / df["ev_annual_sales_10k"],
            np.nan,
        )

    # 消费力指标
    if "urban_income" in df.columns and "gdp_per_capita" in df.columns:
        df["consumption_power_index"] = (
            (df["urban_income"] / df["urban_income"].mean())
            + (df["gdp_per_capita"] / df["gdp_per_capita"].mean())
        ) / 2

    return df


# =============================================================================
# 测试入口
# =============================================================================

if __name__ == "__main__":
    # 快速测试：加载样本数据并运行预处理
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "data"
    raw_path = data_dir / "ev_data_raw.csv"

    if raw_path.exists():
        print("加载原始数据...")
        df_raw = pd.read_csv(raw_path)
        df_clean = preprocess_ev_data(df_raw)
        print("\n📊 各列数据类型:")
        print(df_clean.dtypes)
        print("\n📊 基本统计:")
        print(df_clean[["price_median", "range_km", "battery_capacity",
                         "power_kw", "user_score"]].describe())
    else:
        print(f"❌ 找不到数据文件: {raw_path}")
        print("   请先运行: python generate_sample_data.py")
