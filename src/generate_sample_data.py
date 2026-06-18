"""
=============================================================================
EV-Insight 模拟数据生成器
=============================================================================
由于爬取真实数据存在反爬限制，本脚本可生成高度仿真的新能源汽车
数据集用于开发、测试和演示。

生成的数据字段完全覆盖论文中描述的所有维度：
- 78 个品牌，412 个车系，约 3141 款车型
- 价格区间：3 万 ~ 150 万
- 续航区间：120 km ~ 1000 km
- 包含品牌/车系/车型三级结构

使用方法：
    python generate_sample_data.py
    → 生成 data/ev_data_raw.csv (原始数据)
    → 生成 data/ev_data_cleaned.csv (清洗后数据，可直接用于建模和 Streamlit)
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# 配置
# =============================================================================

# 设置随机种子保证可复现
np.random.seed(42)

# 品牌及其市场定位配置
# (品牌名, 价格基数_万, 定位, 主要车身类型, 是否新势力)
BRAND_CONFIG = [
    # 传统自主品牌
    ("比亚迪", 18, "全价位覆盖", "轿车/SUV/MPV", False),
    ("五菱", 6, "经济型", "微型车/小型车", False),
    ("长安", 15, "中端", "轿车/SUV", False),
    ("吉利", 16, "中端", "轿车/SUV", False),
    ("广汽埃安", 17, "中端", "轿车/SUV", False),
    ("长城欧拉", 14, "中端女性", "小型车/紧凑型", False),
    ("奇瑞", 12, "经济型", "微型车/SUV", False),
    ("上汽荣威", 16, "中端", "轿车/SUV", False),
    ("北汽", 14, "中端", "轿车/SUV", False),
    ("江淮", 10, "经济型", "小型车/MPV", False),
    ("红旗", 38, "高端", "轿车/SUV", False),
    ("东风", 13, "中端", "轿车/SUV", False),
    ("一汽", 15, "中端", "轿车/SUV", False),
    # 造车新势力
    ("蔚来", 45, "高端", "SUV/轿车", True),
    ("小鹏", 22, "中高端", "轿车/SUV", True),
    ("理想", 33, "中高端家庭", "SUV/MPV", True),
    ("问界", 28, "中高端", "SUV", True),
    ("小米汽车", 24, "中高端", "轿车", True),
    ("零跑", 15, "中端性价比", "轿车/SUV", True),
    ("哪吒", 12, "经济型", "SUV/轿车", True),
    ("岚图", 32, "高端", "SUV/MPV", True),
    ("阿维塔", 35, "高端", "SUV/轿车", True),
    ("智己", 30, "中高端", "轿车/SUV", True),
    ("极氪", 28, "中高端", "轿车/SUV", True),
    ("深蓝", 18, "中端", "轿车/SUV", True),
    ("仰望", 85, "超豪华", "SUV/跑车", True),
    # 外资/合资品牌
    ("特斯拉", 32, "中高端", "轿车/SUV", True),
    ("大众ID", 20, "中端", "轿车/SUV", False),
    ("丰田", 19, "中端", "轿车/SUV", False),
    ("本田", 19, "中端", "轿车/SUV", False),
    ("日产", 18, "中端", "轿车/SUV", False),
    ("宝马", 45, "豪华", "轿车/SUV", False),
    ("奔驰", 50, "豪华", "轿车/SUV", False),
    ("奥迪", 42, "豪华", "轿车/SUV", False),
    ("保时捷", 100, "超豪华", "跑车/轿车", False),
    ("沃尔沃", 36, "中高端", "SUV/轿车", False),
    ("凯迪拉克", 38, "中高端", "SUV/轿车", False),
    ("雷克萨斯", 40, "中高端", "轿车/SUV", False),
]

# 车身类型
BODY_TYPES = ["微型车", "小型车", "紧凑型车", "中型车", "中大型车", "大型车",
              "小型SUV", "紧凑型SUV", "中型SUV", "中大型SUV", "大型SUV",
              "MPV", "跑车", "皮卡"]

# 电池类型
BATTERY_TYPES = ["磷酸铁锂", "三元锂", "磷酸铁锂/三元锂混合", "固态电池(半固态)", "钠离子"]

# 能源类型
ENERGY_TYPES = ["纯电动", "插电混动", "增程式"]

# 各省市数据
PROVINCES = [
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
    "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
]

# 智能驾驶等级
ADAS_LEVELS = ["L0", "L1", "L2", "L2+", "L3"]


# =============================================================================
# 车型数据生成
# =============================================================================


def generate_ev_vehicle_data(n_total: int = 3200) -> pd.DataFrame:
    """
    生成仿真的新能源汽车车型数据集。

    Parameters
    ----------
    n_total : int
        目标车型总数（默认 3200，覆盖 3141+ 有效数据）

    Returns
    -------
    pd.DataFrame
        包含所有车型字段的原始数据
    """
    print(f"🔧 正在生成 {n_total} 款新能源车型数据...")

    records = []
    bid = 0  # 品牌 ID

    # 按品牌生成
    for brand_name, price_base, positioning, main_types, is_new_force in BRAND_CONFIG:
        # 每个品牌生成 2-12 个车系
        n_series = np.random.randint(2, 13)
        # 在基础价格上添加 ±40% 的随机偏移
        brand_price_std = price_base * 0.3

        # 品牌级别的电池偏好
        if price_base < 12:
            battery_pref = [0.8, 0.15, 0.03, 0.01, 0.01]  # 偏向铁锂
        elif price_base < 30:
            battery_pref = [0.4, 0.45, 0.08, 0.04, 0.03]
        else:
            battery_pref = [0.1, 0.5, 0.1, 0.2, 0.1]  # 偏向三元/固态

        # 品牌级别的能源类型偏好
        if brand_name == "理想":
            energy_pref = [0.0, 0.0, 1.0]  # 全增程
        elif brand_name == "比亚迪":
            energy_pref = [0.5, 0.35, 0.15]
        elif price_base < 8:
            energy_pref = [0.9, 0.08, 0.02]
        else:
            energy_pref = [0.65, 0.25, 0.10]

        for s in range(n_series):
            series_name = f"{brand_name}-系列{s+1}"
            # 每个车系 3-15 个配置
            n_models = np.random.randint(3, 16)
            series_price_shift = np.random.normal(0, brand_price_std * 0.3)

            for m in range(n_models):
                # ---- 价格 ----
                guide_price = max(2.8, price_base + series_price_shift
                                  + np.random.normal(0, brand_price_std * 0.5))
                dealer_price = guide_price * np.random.uniform(0.92, 1.0)

                # ---- 续航 ----
                # 价格越高，续航越长，但有天花板效应
                base_range = 150 + guide_price * 10
                if guide_price > 10:
                    base_range += (guide_price - 10) * 3
                range_km = int(np.clip(
                    base_range + np.random.normal(0, 40),
                    100, 1050
                ))

                # ---- 电池 ----
                battery_type = np.random.choice(BATTERY_TYPES, p=battery_pref)
                # 容量与续航强相关
                efficiency = np.random.uniform(5.5, 7.5)  # km/kWh
                battery_capacity = round(range_km / efficiency + np.random.normal(0, 3), 1)
                battery_capacity = max(9, min(150, battery_capacity))

                # ---- 动力 ----
                # 功率与价格正相关
                power_kw = int(np.clip(30 + guide_price * 5 + np.random.normal(0, 20), 20, 600))
                torque_nm = int(power_kw * np.random.uniform(1.8, 3.2))
                # 加速：贵车更快
                accel = round(np.clip(12 - guide_price * 0.1 + np.random.normal(0, 0.8), 1.8, 14.0), 1)

                # ---- 车身 ----
                if guide_price < 8:
                    body_type = np.random.choice(["微型车", "小型车", "微型车", "小型车", "紧凑型车"])
                elif guide_price < 15:
                    body_type = np.random.choice(["紧凑型车", "小型SUV", "紧凑型SUV", "中型车"])
                elif guide_price < 30:
                    body_type = np.random.choice(["中型车", "紧凑型SUV", "中型SUV", "中大型车", "MPV"])
                elif guide_price < 50:
                    body_type = np.random.choice(["中大型车", "中型SUV", "中大型SUV", "MPV", "跑车"])
                else:
                    body_type = np.random.choice(["大型车", "中大型SUV", "大型SUV", "跑车", "MPV"])

                # ---- 尺寸（基于车身类型） ----
                size_map = {
                    "微型车": (2900, 1550, 1520, 2000),
                    "小型车": (3700, 1700, 1520, 2450),
                    "紧凑型车": (4500, 1800, 1470, 2680),
                    "中型车": (4850, 1850, 1460, 2850),
                    "中大型车": (5050, 1920, 1480, 3000),
                    "大型车": (5300, 1980, 1500, 3150),
                    "小型SUV": (4200, 1780, 1620, 2580),
                    "紧凑型SUV": (4550, 1850, 1680, 2720),
                    "中型SUV": (4800, 1920, 1720, 2850),
                    "中大型SUV": (5050, 1980, 1780, 2980),
                    "大型SUV": (5250, 2030, 1850, 3120),
                    "MPV": (4900, 1920, 1780, 2980),
                    "跑车": (4600, 1950, 1250, 2750),
                    "皮卡": (5400, 1980, 1880, 3220),
                }
                base_l, base_w, base_h, base_wb = size_map.get(body_type, size_map["中型车"])
                length_mm = int(base_l + np.random.normal(0, 80))
                width_mm = int(base_w + np.random.normal(0, 40))
                height_mm = int(base_h + np.random.normal(0, 30))
                wheelbase_mm = int(base_wb + np.random.normal(0, 60))

                # ---- 智能驾驶 ----
                if guide_price < 10:
                    adas_weights = [0.35, 0.40, 0.20, 0.04, 0.01]
                elif guide_price < 20:
                    adas_weights = [0.05, 0.20, 0.50, 0.20, 0.05]
                elif guide_price < 40:
                    adas_weights = [0.01, 0.05, 0.35, 0.40, 0.19]
                else:
                    adas_weights = [0.0, 0.02, 0.15, 0.40, 0.43]
                adas_level = np.random.choice(ADAS_LEVELS, p=adas_weights)

                # ---- 用户评分 ----
                user_score = round(np.clip(
                    3.2 + (guide_price / 50) * 1.5 + np.random.normal(0, 0.4),
                    2.0, 5.0), 1)
                review_count = int(np.random.exponential(200) + np.random.randint(1, 500))

                # ---- 能源类型 ----
                energy_type = np.random.choice(ENERGY_TYPES, p=energy_pref)

                # ---- 座位数 ----
                if body_type in ["MPV"]:
                    seats = np.random.choice([6, 7])
                elif body_type in ["跑车"]:
                    seats = np.random.choice([2, 4])
                elif body_type in ["微型车"]:
                    seats = np.random.choice([2, 4])
                else:
                    seats = 5

                model_name = f"{series_name} {' '.join([str(np.random.randint(100, 999))])}"
                if energy_type == "纯电动":
                    model_name += " EV"
                elif energy_type == "插电混动":
                    model_name += " PHEV"
                else:
                    model_name += " EREV"

                records.append({
                    "brand": brand_name,
                    "series": series_name,
                    "model": model_name,
                    "guide_price_str": f"{guide_price:.2f}万",
                    "guide_price": round(guide_price, 2),
                    "dealer_price": round(dealer_price, 2),
                    "range_km": range_km,
                    "battery_type": battery_type,
                    "battery_capacity": battery_capacity,
                    "power_kw": power_kw,
                    "torque_nm": torque_nm,
                    "accel_0_100": accel,
                    "body_type": body_type,
                    "length_mm": length_mm,
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "wheelbase_mm": wheelbase_mm,
                    "adas_level": adas_level,
                    "user_score": user_score,
                    "review_count": review_count,
                    "energy_type": energy_type,
                    "seats": seats,
                })

    df = pd.DataFrame(records)

    # 裁剪到目标数量附近
    if len(df) > n_total:
        df = df.sample(n_total, random_state=42).reset_index(drop=True)

    print(f"✅ 生成完成: {len(df)} 款车型, {df['brand'].nunique()} 个品牌, "
          f"{df['series'].nunique()} 个车系")
    return df


# =============================================================================
# 区域经济数据生成
# =============================================================================


def generate_regional_data() -> pd.DataFrame:
    """
    生成全国 31 个省市的区域经济和基础设施数据。

    Returns
    -------
    pd.DataFrame
    """
    print("🔧 正在生成区域经济数据...")

    np.random.seed(123)

    records = []
    for province in PROVINCES:
        # 东部沿海经济较发达
        if province in ["上海", "北京", "天津", "浙江", "江苏", "广东", "福建"]:
            gdp_scale = np.random.uniform(4, 14)  # 万亿
            income_scale = np.random.uniform(5, 9)  # 万元
            charger_density = np.random.uniform(30, 80)  # 公共桩/万人
            ev_sales_scale = np.random.uniform(15, 45)  # 万辆
            policy_level = np.random.choice([2, 3, 3, 3])  # 限牌城市政策力度高
        # 中部
        elif province in ["湖北", "湖南", "河南", "安徽", "江西", "山西", "陕西", "四川", "重庆"]:
            gdp_scale = np.random.uniform(2, 6)
            income_scale = np.random.uniform(3, 5.5)
            charger_density = np.random.uniform(8, 25)
            ev_sales_scale = np.random.uniform(3, 18)
            policy_level = np.random.choice([1, 2, 2])
        # 东北
        elif province in ["辽宁", "吉林", "黑龙江"]:
            gdp_scale = np.random.uniform(1, 3)
            income_scale = np.random.uniform(2.5, 4.5)
            charger_density = np.random.uniform(3, 12)
            ev_sales_scale = np.random.uniform(0.5, 5)
            policy_level = np.random.choice([0, 1, 1])
        # 西部
        else:
            gdp_scale = np.random.uniform(0.3, 3)
            income_scale = np.random.uniform(2, 4.5)
            charger_density = np.random.uniform(1, 12)
            ev_sales_scale = np.random.uniform(0.1, 5)
            policy_level = np.random.choice([0, 1, 1, 2])

        # 冬季平均温度（影响纯电接受度）
        if province in ["黑龙江", "吉林", "辽宁", "内蒙古", "新疆", "青海", "西藏"]:
            winter_temp = np.random.uniform(-25, -5)
        elif province in ["北京", "天津", "河北", "山西", "宁夏", "甘肃", "陕西"]:
            winter_temp = np.random.uniform(-8, 2)
        elif province in ["山东", "河南", "江苏", "安徽", "湖北"]:
            winter_temp = np.random.uniform(-2, 8)
        else:
            winter_temp = np.random.uniform(5, 22)

        # 限牌城市标记
        is_plate_restricted = province in ["北京", "上海", "广州", "深圳", "天津", "杭州"]

        records.append({
            "province": province,
            "gdp_trillion": round(gdp_scale, 2),
            "gdp_per_capita": round(income_scale * 1.4, 1),
            "urban_income": round(income_scale, 1),
            "charger_count_per_10k": round(charger_density, 1),
            "ev_annual_sales_10k": round(ev_sales_scale, 1),
            "policy_level": policy_level,
            "winter_avg_temp": round(winter_temp, 1),
            "is_plate_restricted": int(is_plate_restricted),
        })

    df = pd.DataFrame(records)
    print(f"✅ 区域数据生成完成: {len(df)} 个省市")
    return df


# =============================================================================
# 时间序列数据生成（用于趋势分析）
# =============================================================================


def generate_trend_data() -> pd.DataFrame:
    """
    生成 2020-2025 年的市场趋势数据。

    Returns
    -------
    pd.DataFrame
    """
    print("🔧 正在生成趋势数据...")

    years = list(range(2020, 2026))
    records = []

    for year in years:
        # 逐年增长
        base_models = {2020: 800, 2021: 1300, 2022: 1800, 2023: 2400, 2024: 2900, 2025: 3141}
        avg_range = {2020: 408, 2021: 435, 2022: 475, 2023: 520, 2024: 560, 2025: 592}
        range_std = {2020: 89, 2021: 105, 2022: 125, 2023: 148, 2024: 165, 2025: 178}
        avg_price = {2020: 22.5, 2021: 23.2, 2022: 24.8, 2023: 24.0, 2024: 23.2, 2025: 22.8}
        market_penetration = {2020: 5.8, 2021: 13.4, 2022: 25.6, 2023: 31.6, 2024: 40.2, 2025: 48.5}
        new_force_share = {2020: 4.8, 2021: 8.2, 2022: 12.5, 2023: 16.3, 2024: 19.8, 2025: 21.3}

        records.append({
            "year": year,
            "total_models": base_models[year],
            "avg_range_km": avg_range[year],
            "range_std_km": range_std[year],
            "avg_price_wan": avg_price[year],
            "market_penetration_pct": market_penetration[year],
            "new_force_market_share_pct": new_force_share[year],
        })

    df = pd.DataFrame(records)
    print(f"✅ 趋势数据生成完成: {len(df)} 年")
    return df


# =============================================================================
# 主程序
# =============================================================================


def main():
    """生成所有模拟数据并保存"""
    # 确定数据目录
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EV-Insight 模拟数据生成器")
    print("=" * 60)
    print()

    # ---- 1. 车型数据 ----
    df_vehicles = generate_ev_vehicle_data(n_total=3200)

    # 人为制造一些缺失值（模拟真实数据的特征，约 2-3%）
    mask_price = np.random.random(len(df_vehicles)) < 0.023
    df_vehicles.loc[mask_price, "guide_price"] = np.nan

    mask_range = np.random.random(len(df_vehicles)) < 0.011
    df_vehicles.loc[mask_range, "range_km"] = np.nan

    mask_score = np.random.random(len(df_vehicles)) < 0.03
    df_vehicles.loc[mask_score, "user_score"] = np.nan

    raw_path = data_dir / "ev_data_raw.csv"
    df_vehicles.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"📁 原始车型数据已保存: {raw_path}")

    # ---- 2. 清洗并保存 ----
    from preprocess import preprocess_ev_data

    df_cleaned = preprocess_ev_data(df_vehicles)
    cleaned_path = data_dir / "ev_data_cleaned.csv"
    df_cleaned.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    print(f"📁 清洗后车型数据已保存: {cleaned_path}")
    print(f"   字段数: {len(df_cleaned.columns)}, 记录数: {len(df_cleaned)}")

    # ---- 3. 区域数据 ----
    df_region = generate_regional_data()
    region_path = data_dir / "regional_data.csv"
    df_region.to_csv(region_path, index=False, encoding="utf-8-sig")
    print(f"📁 区域经济数据已保存: {region_path}")

    # ---- 4. 趋势数据 ----
    df_trend = generate_trend_data()
    trend_path = data_dir / "trend_data.csv"
    df_trend.to_csv(trend_path, index=False, encoding="utf-8-sig")
    print(f"📁 趋势数据已保存: {trend_path}")

    # ---- 5. 数据概览 ----
    print()
    print("=" * 60)
    print("📊 数据概览")
    print("=" * 60)
    print(f"  品牌数:        {df_cleaned['brand'].nunique()}")
    print(f"  车系数:        {df_cleaned['series'].nunique()}")
    print(f"  车型数:        {len(df_cleaned)}")
    print(f"  价格区间:      {df_cleaned['price_median'].min():.1f} ~ "
          f"{df_cleaned['price_median'].max():.1f} 万")
    print(f"  平均价格:      {df_cleaned['price_median'].mean():.1f} 万")
    print(f"  续航区间:      {df_cleaned['range_km'].min():.0f} ~ "
          f"{df_cleaned['range_km'].max():.0f} km")
    print(f"  平均续航:      {df_cleaned['range_km'].mean():.0f} km")
    print()
    print("✅ 所有数据已生成完毕，可以运行:")
    print("   python src/model.py          # 训练模型")
    print("   streamlit run app/app.py     # 启动 Web 应用")


if __name__ == "__main__":
    main()
