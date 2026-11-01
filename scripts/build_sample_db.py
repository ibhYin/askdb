"""生成样例电商库（data/sample/shop.db）。

数据是**可复现的**（固定随机种子），所以评测分数不会因为重建库而漂移。
真实评测集将在第 2 周接入 BIRD；这一份用于把整条链路先跑通。
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sample" / "shop.db"

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]
LEVELS = ["普通", "银卡", "金卡"]
CATEGORIES = {
    "手机": (1999, 6999),
    "电脑": (3999, 12999),
    "配件": (49, 499),
    "家电": (299, 3999),
    "图书": (29, 129),
}
CHANNELS = ["APP", "小程序", "网页"]
STATUSES = ["已完成", "已完成", "已完成", "已完成", "待发货", "待发货", "已取消"]
PAY_METHODS = ["微信支付", "支付宝", "银行卡"]
REFUND_REASONS = ["七天无理由", "商品破损", "发错货", "质量问题"]
SURNAMES = list("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许")
GIVEN = ["伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "艳", "杰", "涛", "明", "超", "秀英", "霞"]

SCHEMA = """
CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY,
  name        TEXT NOT NULL,
  city        TEXT NOT NULL,
  level       TEXT NOT NULL,
  signup_date TEXT NOT NULL
);

CREATE TABLE products (
  product_id INTEGER PRIMARY KEY,
  name       TEXT NOT NULL,
  category   TEXT NOT NULL,
  price      REAL NOT NULL,
  cost       REAL NOT NULL
);

CREATE TABLE orders (
  order_id    INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
  order_date  TEXT NOT NULL,
  status      TEXT NOT NULL,
  channel     TEXT NOT NULL
);

CREATE TABLE order_items (
  item_id    INTEGER PRIMARY KEY,
  order_id   INTEGER NOT NULL REFERENCES orders(order_id),
  product_id INTEGER NOT NULL REFERENCES products(product_id),
  quantity   INTEGER NOT NULL,
  unit_price REAL NOT NULL
);

CREATE TABLE payments (
  payment_id INTEGER PRIMARY KEY,
  order_id   INTEGER NOT NULL REFERENCES orders(order_id),
  amount     REAL NOT NULL,
  method     TEXT NOT NULL,
  paid_at    TEXT NOT NULL
);

CREATE TABLE refunds (
  refund_id  INTEGER PRIMARY KEY,
  order_id   INTEGER NOT NULL REFERENCES orders(order_id),
  amount     REAL NOT NULL,
  reason     TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


def build(db_path: Path = DB_PATH) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rng = random.Random(42)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    customers = []
    for cid in range(1, 121):
        name = rng.choice(SURNAMES) + rng.choice(GIVEN)
        customers.append(
            (
                cid,
                name,
                rng.choice(CITIES),
                rng.choices(LEVELS, weights=[6, 3, 1])[0],
                f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            )
        )
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    products = []
    for pid in range(1, 41):
        category = rng.choice(list(CATEGORIES))
        low, high = CATEGORIES[category]
        price = float(rng.randrange(low, high, 10))
        products.append(
            (pid, f"{category}{pid:02d}", category, price, round(price * rng.uniform(0.55, 0.8), 2))
        )
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    price_by_product = {p[0]: p[3] for p in products}
    orders, items, payments, refunds = [], [], [], []
    item_id = payment_id = refund_id = 0

    for oid in range(1, 601):
        customer_id = rng.randint(1, 120)
        order_date = f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        status = rng.choice(STATUSES)
        channel = rng.choice(CHANNELS)
        orders.append((oid, customer_id, order_date, status, channel))

        total = 0.0
        for _ in range(rng.randint(1, 4)):
            item_id += 1
            pid = rng.randint(1, 40)
            qty = rng.randint(1, 3)
            unit_price = price_by_product[pid]
            total += qty * unit_price
            items.append((item_id, oid, pid, qty, unit_price))

        if status != "已取消":
            payment_id += 1
            payments.append(
                (payment_id, oid, round(total, 2), rng.choice(PAY_METHODS), order_date)
            )
        if status == "已完成" and rng.random() < 0.08:
            refund_id += 1
            ratio = rng.choice([1.0, 0.5, 0.3])
            refunds.append(
                (
                    refund_id,
                    oid,
                    round(total * ratio, 2),
                    rng.choice(REFUND_REASONS),
                    order_date,
                )
            )

    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
    conn.executemany("INSERT INTO payments VALUES (?,?,?,?,?)", payments)
    conn.executemany("INSERT INTO refunds VALUES (?,?,?,?,?)", refunds)
    conn.commit()
    conn.close()
    return db_path


def main() -> None:
    path = build()
    conn = sqlite3.connect(path)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    print(f"已生成 {path}")
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} 行")


if __name__ == "__main__":
    main()
