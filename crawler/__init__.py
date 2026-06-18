"""
=============================================================================
EV-Insight 爬虫模块 - 汽车之家 & 懂车帝数据采集
=============================================================================
本模块实现了文档 2.2 节描述的 "Scrapy + Selenium" 双轨爬虫架构。

⚠️ 重要提示：
  当前汽车之家、懂车帝等平台的反爬机制极其严格。
  直接运行本脚本大概率会被封 IP。
  建议方案：
    A) 配置代理 IP 池 + 大幅降低请求频率（每次请求间隔 5-10 秒）
    B) 使用 Kaggle/天池等平台下载现成的新能源汽车数据集
    C) 运行 src/generate_sample_data.py 生成模拟数据用于学习和演示

使用方法（谨慎）：
    cd crawler/
    scrapy runspider autohome_spider.py -o ../data/raw_ev_data.json
=============================================================================
"""

import time
import random
import json
import logging
from datetime import datetime

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import Request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import pandas as pd

# =============================================================================
# 反爬策略配置
# =============================================================================

# User-Agent 轮换池（模拟不同浏览器和设备）
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# 随机请求间隔（秒）- 非常重要！
MIN_DELAY = 3.0
MAX_DELAY = 8.0

# =============================================================================
# Selenium 驱动初始化
# =============================================================================


def create_webdriver(headless: bool = True) -> webdriver.Chrome:
    """
    创建配置好的 Chrome WebDriver 实例。

    Parameters
    ----------
    headless : bool
        是否使用无头模式（服务器部署时建议 True）

    Returns
    -------
    webdriver.Chrome
    """
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


# =============================================================================
# 辅助函数
# =============================================================================


def parse_price(price_str: str) -> tuple:
    """
    解析价格字符串，如 "14.98-19.98万" → (14.98, 19.98)

    Parameters
    ----------
    price_str : str
        原始价格字符串

    Returns
    -------
    tuple[float, float] 或 (None, None)
    """
    import re

    if not price_str or not isinstance(price_str, str):
        return None, None

    # 匹配 "14.98-19.98万" 或 "14.98万" 或 "14.98-19.98万元"
    pattern = r"([\d.]+)\s*(?:-|~)\s*([\d.]+)\s*万"
    match = re.search(pattern, price_str)
    if match:
        return float(match.group(1)), float(match.group(2))

    # 单价格 "14.98万"
    single_pattern = r"([\d.]+)\s*万"
    match = re.search(single_pattern, price_str)
    if match:
        price = float(match.group(1))
        return price, price

    return None, None


def safe_extract(element, selector: str, default: str = ""):
    """安全提取元素文本"""
    try:
        result = element.css(selector).get()
        return result.strip() if result else default
    except Exception:
        return default


# =============================================================================
# Scrapy Spider - 汽车之家
# =============================================================================


class AutoHomeEVSpider(scrapy.Spider):
    """
    汽车之家新能源汽车数据采集 Spider。

    采集字段：
    - brand:              品牌
    - series:             车系
    - model:              车型
    - guide_price:        官方指导价
    - dealer_price:       经销商参考价
    - range_km:           续航里程 (km)
    - battery_type:       电池类型
    - battery_capacity:   电池容量 (kWh)
    - power_kw:           电机功率 (kW)
    - torque_nm:          扭矩 (Nm)
    - accel_0_100:        百公里加速 (s)
    - body_type:          车身类型
    - length_mm:          车长 (mm)
    - width_mm:           车宽 (mm)
    - height_mm:          车高 (mm)
    - wheelbase_mm:       轴距 (mm)
    - adas_level:         智能驾驶等级
    - user_score:         用户评分
    - review_count:       评价数量
    - energy_type:        能源类型 (纯电/插混/增程)
    - seats:              座位数
    """

    name = "autohome_ev"
    allowed_domains = ["autohome.com.cn"]

    # 汽车之家新能源汽车筛选页
    start_urls = [
        "https://www.autohome.com.cn/grade/carhtml/NE.html",
    ]

    custom_settings = {
        "DOWNLOAD_DELAY": random.uniform(MIN_DELAY, MAX_DELAY),
        "CONCURRENT_REQUESTS": 1,  # 单线程，防止被封
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "USER_AGENT": random.choice(USER_AGENTS),
        "ROBOTSTXT_OBEY": False,
        "COOKIES_ENABLED": True,
        "RETRY_TIMES": 3,
        "DOWNLOAD_TIMEOUT": 30,
        # 自动限速扩展
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 5,
        "AUTOTHROTTLE_MAX_DELAY": 15,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 0.5,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver = None
        self.items_scraped = 0
        logging.info("AutoHomeEVSpider 初始化完成")

    def parse(self, response):
        """
        解析品牌和车系列表页面。
        """
        self.logger.info(f"正在解析页面: {response.url}")

        # 尝试从 JSON 接口获取数据（更高效）
        # 汽车之家通常有隐藏的 API 接口返回结构化 JSON
        brands_data = self._try_parse_json_api(response)

        if brands_data:
            yield from self._process_brands_from_json(brands_data)
        else:
            # 回退到 HTML 解析
            yield from self._process_brands_from_html(response)

    def _try_parse_json_api(self, response):
        """尝试从内嵌 JSON 或 API 接口获取品牌数据"""
        try:
            # 查找页面内嵌的初始化数据
            script_text = response.css(
                "script:contains('window.__INITIAL_STATE__')"
            ).get()
            if script_text:
                import re

                match = re.search(
                    r"window\.__INITIAL_STATE__\s*=\s*({.*?});", script_text, re.DOTALL
                )
                if match:
                    return json.loads(match.group(1))
        except Exception as e:
            self.logger.warning(f"JSON 解析失败: {e}")
        return None

    def _process_brands_from_json(self, brands_data):
        """处理 JSON 格式的品牌数据"""
        brands = brands_data.get("brandList", [])
        for brand in brands:
            brand_name = brand.get("name", "")
            series_list = brand.get("seriesList", [])
            for series in series_list:
                series_name = series.get("name", "")
                series_id = series.get("id", "")
                series_url = f"https://www.autohome.com.cn/{series_id}/"
                time.sleep(random.uniform(1, 3))
                yield Request(
                    series_url,
                    callback=self.parse_series,
                    meta={"brand": brand_name, "series": series_name},
                )

    def _process_brands_from_html(self, response):
        """处理 HTML 格式的品牌列表"""
        brand_elements = response.css("dl.list-dl")
        for brand_el in brand_elements:
            brand_name = safe_extract(brand_el, "dt a::text")
            if not brand_name:
                continue
            series_links = brand_el.css("dd a")
            for link in series_links:
                series_name = safe_extract(link, "::text")
                series_url = link.css("::attr(href)").get()
                if series_url:
                    time.sleep(random.uniform(1, 3))
                    yield response.follow(
                        series_url,
                        callback=self.parse_series,
                        meta={"brand": brand_name, "series": series_name},
                    )

    def parse_series(self, response):
        """
        解析车系详情页，提取车型列表。
        对于动态渲染的页面，使用 Selenium 辅助。
        """
        brand = response.meta["brand"]
        series = response.meta["series"]

        # 尝试获取车型配置 JSON（通常有 API 接口）
        # URL 模式: https://www.autohome.com.cn/ashx/series_allspec.ashx?s={series_id}
        self.logger.info(f"正在解析车系: {brand} - {series}")

        # 提取车型列表（简化版 - 实际需要根据页面结构调整选择器）
        model_items = response.css("div.model-list li, tr.model-item")
        for item in model_items:
            item_data = self._extract_model_data(item, brand, series)
            if item_data:
                self.items_scraped += 1
                yield item_data

    def _extract_model_data(self, item, brand: str, series: str) -> dict:
        """从单个车型元素中提取所有字段"""
        model_name = safe_extract(item, ".model-name::text, td:nth-child(1)::text")
        if not model_name:
            return None

        guide_price_str = safe_extract(
            item, ".guide-price::text, td:nth-child(2)::text"
        )
        price_low, price_high = parse_price(guide_price_str)

        return {
            "brand": brand,
            "series": series,
            "model": model_name,
            "guide_price": guide_price_str,
            "price_low": price_low,
            "price_high": price_high,
            "dealer_price": safe_extract(item, ".dealer-price::text"),
            "range_km": safe_extract(item, ".range::text, td:nth-child(3)::text"),
            "battery_type": safe_extract(item, ".battery::text"),
            "battery_capacity": safe_extract(item, ".capacity::text"),
            "power_kw": safe_extract(item, ".power::text"),
            "torque_nm": safe_extract(item, ".torque::text"),
            "accel_0_100": safe_extract(item, ".accel::text"),
            "body_type": safe_extract(item, ".body-type::text"),
            "length_mm": safe_extract(item, ".length::text"),
            "width_mm": safe_extract(item, ".width::text"),
            "height_mm": safe_extract(item, ".height::text"),
            "wheelbase_mm": safe_extract(item, ".wheelbase::text"),
            "adas_level": safe_extract(item, ".adas::text"),
            "user_score": safe_extract(item, ".score::text"),
            "review_count": safe_extract(item, ".review-count::text"),
            "energy_type": safe_extract(item, ".energy::text"),
            "seats": safe_extract(item, ".seats::text"),
            "crawl_time": datetime.now().isoformat(),
        }

    def closed(self, reason):
        """Spider 关闭时的回调"""
        if self.driver:
            self.driver.quit()
        logging.info(
            f"Spider 已关闭 (原因: {reason})，共采集 {self.items_scraped} 条数据"
        )


# =============================================================================
# 懂车帝补充数据采集
# =============================================================================


class DongCheDiSpider(scrapy.Spider):
    """
    懂车帝新能源汽车补充数据采集 Spider。

    重点关注：
    - 用户口碑标签（"续航扎实""智能化高""空间大"等）
    - 车主真实续航反馈（冬季续航达成率）
    - 官方实测数据（实测续航、实测加速）
    """

    name = "dongchedi_ev"
    allowed_domains = ["dongchedi.com"]
    start_urls = ["https://www.dongchedi.com/auto/library/x-x-x-x-x-x-x-x-x"]

    custom_settings = {
        "DOWNLOAD_DELAY": random.uniform(3, 7),
        "CONCURRENT_REQUESTS": 1,
        "USER_AGENT": random.choice(USER_AGENTS),
        "ROBOTSTXT_OBEY": False,
    }

    def parse(self, response):
        """解析车型库页面"""
        # 提取车型链接
        car_links = response.css("a[href*='/auto/']::attr(href)").getall()
        for link in car_links[:50]:  # 限制数量用于测试
            yield response.follow(link, self.parse_car_detail)

    def parse_car_detail(self, response):
        """解析车型详情页"""
        car_name = response.css("h1::text").get("").strip()
        score = response.css(".score-value::text").get("")
        tags = response.css(".tag-item::text").getall()

        yield {
            "car_name": car_name,
            "url": response.url,
            "user_score": score,
            "user_tags": ",".join(tags),
            "crawl_time": datetime.now().isoformat(),
        }


# =============================================================================
# 独立运行入口（不使用 Scrapy 命令行时）
# =============================================================================


def run_autohome_spider(output_path: str = "../data/raw_ev_data.json"):
    """
    以编程方式运行汽车之家爬虫。

    Parameters
    ----------
    output_path : str
        输出文件路径
    """
    process = CrawlerProcess(
        settings={
            "FEEDS": {output_path: {"format": "json", "encoding": "utf-8"}},
            "LOG_LEVEL": "INFO",
        }
    )
    process.crawl(AutoHomeEVSpider)
    process.start()

    # 加载结果
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"\n✅ 爬取完成！共采集 {len(data)} 条车型数据")
        print(f"📁 数据已保存至: {output_path}")
        return data
    except FileNotFoundError:
        print("❌ 爬取失败，未生成数据文件")
        return []


def run_dongchedi_spider(output_path: str = "../data/raw_dcd_data.json"):
    """运行懂车帝爬虫"""
    process = CrawlerProcess(
        settings={
            "FEEDS": {output_path: {"format": "json", "encoding": "utf-8"}},
            "LOG_LEVEL": "INFO",
        }
    )
    process.crawl(DongCheDiSpider)
    process.start()


# =============================================================================
# Selenium 辅助采集函数（用于动态渲染页面）
# =============================================================================


def selenium_fetch_dynamic_page(url: str, wait_selector: str, timeout: int = 15) -> str:
    """
    使用 Selenium 获取动态渲染页面的完整 HTML。

    Parameters
    ----------
    url : str
        目标页面 URL
    wait_selector : str
        等待渲染完成的 CSS 选择器
    timeout : int
        最大等待时间（秒）

    Returns
    -------
    str
        页面渲染后的完整 HTML
    """
    driver = None
    try:
        driver = create_webdriver(headless=True)
        driver.get(url)

        # 等待关键元素渲染完成
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
        )

        # 模拟人类行为：随机滚动
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(random.uniform(0.5, 1.5))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(0.5, 1.0))

        return driver.page_source
    except Exception as e:
        logging.error(f"Selenium 获取页面失败 [{url}]: {e}")
        return ""
    finally:
        if driver:
            driver.quit()


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EV-Insight 数据采集模块")
    print("=" * 60)
    print()
    print("⚠️  警告：直接爬取汽车之家/懂车帝可能触发反爬机制。")
    print("   建议先运行以下命令生成模拟数据用于开发测试：")
    print("   python ../src/generate_sample_data.py")
    print()
    print("如确认要运行爬虫，请取消下方注释：")
    print()
    print("  # 运行汽车之家爬虫")
    print("  run_autohome_spider('../data/raw_ev_data.json')")
    print()
    print("  # 运行懂车帝爬虫")
    print("  run_dongchedi_spider('../data/raw_dcd_data.json')")
