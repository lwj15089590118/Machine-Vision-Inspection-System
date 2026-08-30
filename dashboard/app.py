# -*- coding: utf-8 -*-
"""
dashboard/app.py —— 生产看板服务（Flask + ECharts）
================================================================
职责：
    只读消费 plc_link/plc_server.py 落盘的三类文件，向前端单页提供
    轮询接口；并支持手动触发（看板以 Modbus 客户端身份写 HR0=1，
    与真实产线上位机按钮走同一通路）：
      data/records.jsonl      检测记录流（JSON Lines，追加写）
      data/annot/latest.png   最新标注帧（定位框+缺陷红框+结论文字）
      data/annot/latest.json  最新检测结果 JSON

接口：
    GET  /                看板单页（templates/index.html，ECharts CDN）
    GET  /annot/<name>    标注图等静态文件（禁缓存，配合时间戳刷新）
    GET  /api/state       轮询接口（约 1.5s 一次）：最新帧信息 + 统计
                         汇总 + 最近 50 条记录 + 节拍序列（图表数据）
    POST /api/trigger     写 HR0=1 触发一次检测（需 plc_server 在运行）

统计口径：
    - 良率 = OK / (OK + NG)，FAULT（看门狗故障）不计入良率分母；
    - NG 类型分布对每条 NG 记录的 defect_types 逐类型计数；
    - 节拍曲线取最近 120 条非故障记录的 duration_ms。

命令行用法：
    python dashboard/app.py            # 先启动 plc_link/plc_server.py 更佳
    浏览器打开 http://127.0.0.1:5001（端口取自 config.DASH_PORT，
    若与其他开发服务冲突可直接改 config）

说明：
    - 看板与视觉服务解耦：plc_server 未运行时看板仍可打开（展示历史
      数据），仅"手动触发"会返回连接错误提示；
    - records.jsonl 由服务端持续追加，本模块每次全量读取后取尾部——
      演示数据量（数千条内）下开销可忽略，工程上可改为按字节偏移增量读。
"""
import json
import time
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, render_template, send_file
from pymodbus.client import ModbusTcpClient

# 允许直接 `python dashboard/app.py` 运行：把项目根目录加入 sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

app = Flask(__name__)

RECORD_TAIL = 50          # 记录表返回条数
BEAT_SERIES = 120         # 节拍曲线点数


# ----------------------------------------------------------------
# 数据读取
# ----------------------------------------------------------------
def read_records(limit: int = None) -> list:
    """
    读取 records.jsonl（尾部 limit 条）。文件不存在或行为空/半行
    （服务端正在写入）时跳过，保证轮询永不抛异常。
    """
    path = config.DATA_DIR / "records.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                      # 半行/坏行直接丢弃
    return out[-limit:] if limit else out


def read_latest() -> dict:
    """读取最新结果 JSON 与标注图的修改时间（不存在则返回占位结构）"""
    png = config.ANNOT_DIR / "latest.png"
    js = config.ANNOT_DIR / "latest.json"
    info = {"image_exists": png.exists(), "mtime": None, "result": None}
    if png.exists():
        info["mtime"] = int(png.stat().st_mtime)
    if js.exists():
        try:
            info["result"] = json.loads(js.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return info


def compute_stats(records: list) -> dict:
    """汇总统计：良率 / NG 类型分布 / 节拍序列"""
    total = len(records)
    faults = sum(1 for r in records if r.get("fault"))
    ok = sum(1 for r in records if r.get("result") == "OK")
    ng = sum(1 for r in records if r.get("result") == "NG")
    judged = ok + ng
    yield_pct = round(ok / judged * 100.0, 2) if judged else None

    ng_types = Counter()
    for r in records:
        if r.get("result") == "NG":
            for t in (r.get("defect_types") or []):
                ng_types[t] += 1

    beats = [(r.get("seq"), r.get("duration_ms")) for r in records
             if not r.get("fault") and isinstance(r.get("duration_ms"),
                                                  (int, float))]
    beats = beats[-BEAT_SERIES:]
    avg_beat = round(sum(b for _, b in beats) / len(beats), 1) if beats \
        else None
    max_beat = round(max(b for _, b in beats), 1) if beats else None

    return {"total": total, "ok": ok, "ng": ng, "fault": faults,
            "yield_pct": yield_pct,
            "ng_types": dict(ng_types),
            "beat_seq": [s for s, _ in beats],
            "beat_ms": [b for _, b in beats],
            "avg_beat_ms": avg_beat, "max_beat_ms": max_beat}


def trim_record(r: dict) -> dict:
    """裁剪记录字段（表格展示所需子集）。
    记录的字段契约由 vision_pipeline.build_record 唯一定义且满键同构
    （FAULT 记录亦补齐全部键），此处的 .get 默认值仅作读侧兜底——
    用于兼容历史 jsonl 文件，不再是契约的一部分。"""
    return {"seq": r.get("seq"), "ts": r.get("ts", ""),
            "result": r.get("result", "?"),
            "defect_types": r.get("defect_types") or [],
            "hole_max_offset_mm": r.get("hole_max_offset_mm"),
            "confidence": r.get("confidence"),
            "duration_ms": r.get("duration_ms"),
            "truth_defects": r.get("truth_defects") or []}


# ----------------------------------------------------------------
# 路由
# ----------------------------------------------------------------
@app.route("/")
def index():
    """看板单页（缺陷类型中文名由 config 注册表注入，前端不再自带副本）"""
    return render_template("index.html", type_cn=config.DEFECT_CN)


@app.route("/annot/<path:name>")
def annot_file(name: str):
    """标注图静态文件（禁缓存，前端用 ?t=<mtime> 控制刷新）"""
    path = (config.ANNOT_DIR / name).resolve()
    # 防路径穿越：只允许 ANNOT_DIR 内的文件（Path 相对关系判断，
    # 替代旧的字符串前缀比较——同级同名前缀目录可绕过前缀检查）
    if not path.is_relative_to(config.ANNOT_DIR.resolve()) or \
            not path.exists():
        return jsonify({"error": "file not found"}), 404
    resp = send_file(str(path), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.route("/api/state")
def api_state():
    """轮询主接口：最新帧 + 统计 + 最近记录"""
    latest = read_latest()
    records = read_records()
    stats = compute_stats(records)
    return jsonify({
        "latest": {
            "image_exists": latest["image_exists"],
            "mtime": latest["mtime"],
            "image_url": f"/annot/latest.png?t={latest['mtime']}"
                         if latest["image_exists"] else None,
            "result": latest["result"],
        },
        "stats": stats,
        "records": [trim_record(r) for r in reversed(records[-RECORD_TAIL:])],
    })


@app.route("/api/trigger", methods=["POST"])
def api_trigger():
    """
    手动触发一次检测：以 Modbus/TCP 客户端身份向从站写 HR0=1。
    plc_server 未运行时返回 503 与中文提示。
    """
    client = ModbusTcpClient(config.MODBUS_HOST, port=config.MODBUS_PORT,
                             timeout=2.0)
    try:
        if not client.connect():
            return jsonify({"ok": False,
                            "msg": f"无法连接视觉服务 "
                                   f"{config.MODBUS_HOST}:{config.MODBUS_PORT}，"
                                   f"请先运行 python plc_link/plc_server.py"}
                           ), 503
        rr = client.write_register(config.REG_TRIGGER, 1)
        if rr.isError():
            return jsonify({"ok": False, "msg": f"写入触发寄存器失败: {rr}"}), 500
        return jsonify({"ok": True,
                        "msg": "已写入 HR0=1，等待视觉服务完成检测……"})
    finally:
        client.close()


# ----------------------------------------------------------------
# 入口
# ----------------------------------------------------------------
if __name__ == "__main__":
    config.ensure_dirs()
    print(f"[DASHBOARD] 看板服务启动 http://{config.DASH_HOST}:"
          f"{config.DASH_PORT} （数据目录 {config.DATA_DIR}）")
    app.run(host=config.DASH_HOST, port=config.DASH_PORT, debug=False)
