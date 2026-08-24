# -*- coding: utf-8 -*-
"""
plc_link/plc_server.py —— Modbus/TCP 从站（模拟 PLC）+ 视觉服务主循环
================================================================
职责：
    1. 用 pymodbus 起一个 Modbus/TCP 从站，模拟产线 PLC 的保持寄存器区；
    2. 视觉服务主循环：轮询触发寄存器 HR0 → 合成一帧"相机画面" →
       locate 定位 + inspect 检测 → 把结果写回寄存器 → 置完成标志 →
       心跳递增；
    3. 超时看门狗：触发后 WATCHDOG_TIMEOUT_S 秒内未完成 → 写故障码，
       结果作废（上位机读 HR2=999 即知异常）；
    4. 每次检测追加一条记录到 data/records.jsonl，并把最新标注帧写入
       data/annot/latest.png、最新结果写入 data/annot/latest.json，
       供 dashboard 看板读取展示。

寄存器规划（保持寄存器 HR，地址从 0 起，zero_mode）：
    HR0  触发命令：上位机写 1=正常触发；写 2=触发并模拟处理卡死
                    （专用于验证看门狗）；视觉端处理后自动清零
    HR1  忙闲状态：0=空闲 1=忙 2=完成（本次结果有效）
    HR2  检测结果码：0=无结果 1=OK 2=NG 999=看门狗故障
    HR3  缺陷类型码（位组合）：
                    bit0=划痕 bit1=崩边 bit2=污渍 bit3=孔偏移
                    bit4=孔缺失 bit5=定位失败
    HR4  X 定位偏差（0.01mm 单位，有符号 int16）
    HR5  Y 定位偏差（0.01mm 单位，有符号 int16）
    HR6  工件角度（0.1° 单位，有符号 int16）
    HR10 心跳计数（uint16，自然回绕；每完成一次检测 +1，空闲时按
                    HEARTBEAT_PERIOD_S 周期 +1）

用法（终端 A）：
    python plc_link/plc_server.py                    # 默认缺陷率 0.3
    python plc_link/plc_server.py --defect-rate 0.5 --seed 42 --port 502
配套自测（终端 B）：
    python plc_link/modbus_client_test.py

说明：
    - 视觉端读写寄存器直接操作服务端数据存储（与网络读写等效且更快），
      上位机/看板则通过 Modbus/TCP 客户端访问；
    - "写 2 模拟卡死"是给看门狗做的可验证测试入口（正常处理约 0.2s，
      远小于 2s 看门狗，永远不会自然超时）。
"""
import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# 允许直接 `python plc_link/plc_server.py` 运行：项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2

import config
# 注意：不要用 pymodbus.server.StartTcpServer —— 它内部是
# asyncio.run(StartAsyncTcpServer(...))，会阻塞调用线程直到 shutdown，
# 导致视觉服务主循环永远无法启动（症状：Modbus 端口可连、寄存器可读写，
# 但触发无人消费、心跳不递增）。
# 正确做法：专用线程内自建事件循环 → 构造 ModbusTcpServer（其构造器要求
# 已有 running loop）→ await serve_forever()；停止时用
# run_coroutine_threadsafe 调 shutdown() 协程。
from pymodbus.server import ModbusTcpServer
from pymodbus.datastore import (ModbusSequentialDataBlock, ModbusSlaveContext,
                                ModbusServerContext)
from simulator.synth import synth_frame
from locate.locate import locate
import vision_pipeline as vp

# pymodbus 自身日志非常啰嗦，压到 WARNING
logging.getLogger("pymodbus").setLevel(logging.WARNING)


N_REGS = 16                      # 寄存器区长度（HR0~HR15，预留扩展）


class VisionService:
    """Modbus 从站 + 视觉服务主循环（单进程，两个线程）"""

    def __init__(self, host: str, port: int, defect_rate: float, seed: int):
        self.host, self.port = host, port
        self.defect_rate = defect_rate
        self.rng = np.random.default_rng(seed)
        self.seq = 0                          # 检测序号（记录用）
        self.server = None
        self._worker = None                   # 当前处理线程
        self._deadline = 0.0                  # 当前处理的看门狗截止时刻
        self._faulted = False                 # 看门狗是否已判故障（作废结果）
        self._last_hb = time.perf_counter()   # 上次空闲心跳时刻
        self._stop = threading.Event()
        self._lock = threading.Lock()         # 保护 _faulted / 寄存器写
        self._loop = None                     # Modbus 从站的 asyncio 事件循环
        self._ready = threading.Event()       # 从站端口就绪信号

        # Modbus 数据存储：HR0~HR15 全 0 初始；zero_mode=True 使地址 0 即 HR0
        store = ModbusSlaveContext(
            hr=ModbusSequentialDataBlock(0, [0] * N_REGS),
            zero_mode=True)
        self.context = ModbusServerContext(slaves=store, single=True)

        config.ensure_dirs()
        self.records_path = config.DATA_DIR / "records.jsonl"

    # ---------------- 寄存器读写助手（服务端直读直写） ----------------
    def read_reg(self, addr: int) -> int:
        """读单个保持寄存器（function code 3 = holding）"""
        return int(self.context[0].getValues(3, addr, count=1)[0])

    def write_reg(self, addr: int, value: int) -> None:
        """写单个保持寄存器"""
        self.context[0].setValues(3, addr, [int(value) & 0xFFFF])

    def read_regs(self, addr: int, count: int) -> list:
        return [int(v) for v in self.context[0].getValues(3, addr, count)]

    # ---------------- Modbus 从站线程 ----------------
    def start_server(self) -> None:
        """
        启动 Modbus/TCP 从站：在独立守护线程里自建 asyncio 事件循环并
        运行 serve_forever（不阻塞调用方）。阻塞至端口就绪（≤5s）。
        """
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._serve, name="modbus-server",
                         daemon=True).start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("Modbus 从站 5s 内未完成端口绑定")
        print(f"[PLC-SIM] Modbus/TCP 从站已就绪 {self.host}:{self.port}")

    def _serve(self) -> None:
        """从站线程入口：建事件循环 → 构造服务器 → serve_forever"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._build_and_serve())

    async def _build_and_serve(self) -> None:
        """协程：构造服务器（需要 running loop）→ 监听 → 持续服务"""
        self.server = ModbusTcpServer(context=self.context,
                                      address=(self.host, self.port))
        self._ready.set()                     # 构造完成即已绑定端口
        await self.server.serve_forever()

    def stop_server(self) -> None:
        """跨线程调度 shutdown 协程，优雅关闭从站（幂等）"""
        if self.server is None or self._loop is None:
            return

        async def _shutdown():
            await self.server.shutdown()

        try:
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            fut.result(timeout=3.0)
        except Exception as e:                # 关闭失败不影响进程退出
            print(f"[PLC-SIM] 从站关闭异常(忽略): {e!r}")

    # ---------------- 单次视觉处理（工作线程内执行） ----------------
    def _process_once(self, trigger_code: int) -> None:
        """
        合成一帧 → 定位 → 检测 → 把结果写入寄存器。
        trigger_code==2 时先睡过看门狗时限再出结果，模拟处理卡死，
        用于验证看门狗逻辑（真实产线对应：相机取图阻塞/算法死循环）。
        """
        t_start = time.perf_counter()
        if trigger_code == 2:
            time.sleep(config.WATCHDOG_TIMEOUT_S + 1.5)

        with self._lock:
            faulted = self._faulted           # 看门狗已判故障则丢弃结果

        # 1) 合成"相机画面"（按设定缺陷率注入缺陷）
        frame, truth = synth_frame(self.rng, with_defects=True,
                                   defect_rate=self.defect_rate)
        # 2) 定位 + 检测（共享流水线；耗时为纯算法口径）
        result, duration_ms = vp.inspect_frame(frame)

        if faulted:
            return                             # 迟到的结果作废，不写寄存器

        # 3) 结果写回寄存器
        defect_code = vp.defect_code_of(result)
        with self._lock:
            self.write_reg(config.REG_RESULT,
                           config.RESULT_OK if result["result"] == "OK"
                           else config.RESULT_NG)
            self.write_reg(config.REG_DEFECT, defect_code)
            if result["locate"].get("ok"):
                self.write_reg(config.REG_DEV_X,
                               config.to_int16(round(
                                   result["locate"]["center_mm"][0] * 100)))
                self.write_reg(config.REG_DEV_Y,
                               config.to_int16(round(
                                   result["locate"]["center_mm"][1] * 100)))
                self.write_reg(config.REG_ANGLE,
                               config.to_int16(round(
                                   result["locate"]["angle_deg"] * 10)))
            self.write_reg(config.REG_HEARTBEAT,
                           (self.read_reg(config.REG_HEARTBEAT) + 1) & 0xFFFF)
            self.write_reg(config.REG_BUSY, config.BUSY_DONE)

        # 4) 落盘记录 + 最新标注帧（供 dashboard）
        self.seq += 1
        rec = vp.build_record(self.seq, result,
                              duration_ms=duration_ms,
                              truth_types=[d["type"] for d in
                                           truth["defects"]])
        self._append_record(rec)
        self._save_latest(frame, result)
        print(f"[VISION] #{self.seq} {result['result']} "
              f"{result['defect_types'] or ''} "
              f"耗时{duration_ms:.0f}ms 真值={rec['truth_defects'] or 'OK件'}")

    def _append_record(self, rec: dict) -> None:
        """追加一条检测记录（JSON Lines，dashboard 轮询读取）"""
        with open(self.records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _save_latest(self, frame: np.ndarray, result: dict) -> None:
        """保存最新标注帧与结果 JSON（dashboard 展示用；委托共享流水线）"""
        vp.save_latest_assets(frame, result)

    # ---------------- 视觉服务主循环 ----------------
    def run(self) -> None:
        """
        状态机：IDLE（轮询触发+空闲心跳）⇄ BUSY（监测工作线程+看门狗）。
        """
        print(f"[VISION] 视觉服务主循环启动（缺陷率={self.defect_rate}，"
              f"看门狗={config.WATCHDOG_TIMEOUT_S}s，Ctrl+C 退出）")
        try:
            while not self._stop.is_set():
                if self._worker is None:
                    # ---- IDLE：轮询触发 ----
                    trig = self.read_reg(config.REG_TRIGGER)
                    if trig in (1, 2):
                        self._faulted = False
                        self.write_reg(config.REG_TRIGGER, 0)   # 清触发
                        self.write_reg(config.REG_BUSY, config.BUSY_BUSY)
                        self._deadline = time.perf_counter() + \
                            config.WATCHDOG_TIMEOUT_S
                        self._worker = threading.Thread(
                            target=self._worker_main, args=(trig,),
                            name="vision-worker", daemon=True)
                        self._worker.start()
                        continue
                    # 空闲心跳：按周期递增（让上位机确认视觉端活着）
                    now = time.perf_counter()
                    if now - self._last_hb >= config.HEARTBEAT_PERIOD_S:
                        self.write_reg(config.REG_HEARTBEAT,
                                       (self.read_reg(
                                           config.REG_HEARTBEAT) + 1) & 0xFFFF)
                        self._last_hb = now
                    time.sleep(0.01)
                else:
                    # ---- BUSY：等工作线程结束或看门狗超时 ----
                    if self._worker.is_alive():
                        if time.perf_counter() > self._deadline:
                            # 看门狗动作：置故障码+完成标志，作废迟到结果
                            with self._lock:
                                self._faulted = True
                                self.write_reg(config.REG_RESULT,
                                               config.RESULT_FAULT)
                                self.write_reg(config.REG_BUSY,
                                               config.BUSY_DONE)
                            self.seq += 1
                            rec = vp.build_record(
                                self.seq, fault=True,
                                duration_ms=config.WATCHDOG_TIMEOUT_S * 1000)
                            self._append_record(rec)
                            print(f"[WATCHDOG] #{self.seq} 处理超时"
                                  f"（>{config.WATCHDOG_TIMEOUT_S}s），"
                                  f"已写故障码 {config.RESULT_FAULT}")
                            self._worker.join()   # 等工作线程退出（卡死模拟
                                                  # 只睡固定时长，必然结束）
                            self._worker = None
                        else:
                            time.sleep(0.005)
                    else:
                        self._worker = None       # 正常完成（结果已写）
        except KeyboardInterrupt:
            print("\n[VISION] 收到 Ctrl+C，正在退出……")
        finally:
            self.shutdown()

    def _worker_main(self, trigger_code: int) -> None:
        """工作线程入口：异常不得杀死主循环"""
        try:
            self._process_once(trigger_code)
        except Exception as e:
            print(f"[VISION] 处理异常: {e!r}")
            with self._lock:
                if not self._faulted:            # 未被看门狗处理过的异常
                    self.write_reg(config.REG_RESULT, config.RESULT_FAULT)
                    self.write_reg(config.REG_BUSY, config.BUSY_DONE)

    def shutdown(self) -> None:
        self._stop.set()
        self.stop_server()
        print("[PLC-SIM] Modbus 从站已停止，进程退出")


# ================================================================
# 命令行入口
# ================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Modbus/TCP 从站模拟 PLC + 视觉服务主循环",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--host", default=config.MODBUS_HOST, help="监听地址")
    ap.add_argument("--port", type=int, default=config.MODBUS_PORT,
                    help="监听端口")
    ap.add_argument("--defect-rate", type=float, default=0.3,
                    help="每帧注入缺陷的概率（演示用）")
    ap.add_argument("--seed", type=int, default=None,
                    help="合成随机种子（固定后每次触发序列可复现）")
    args = ap.parse_args()

    svc = VisionService(args.host, args.port, args.defect_rate, args.seed)
    svc.start_server()
    svc.run()


if __name__ == "__main__":
    main()
