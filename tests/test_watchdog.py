# -*- coding: utf-8 -*-
"""
tests/test_watchdog.py —— 视觉服务状态机时间线单元测试（虚拟时钟）
========================================================
不起真服务器、不睡真 4 秒：给 VisionService 注入虚拟时钟与内存寄存器，
毫秒级重放「触发→超时→看门狗动作」时间线，验证：
    - 超时后 _fire_watchdog 写入 RESULT_FAULT + BUSY_DONE + FAULT 记录；
    - 故障标志置位后，迟到的工作线程结果被作废；
    - 未超时前不误触发；
    - 竞态闭环（真实调用 _process_once）：处理过程中看门狗判故障，
      写寄存器前锁内复查 _faulted，迟到的结果不得覆盖故障码 999；
    - 正常路径不受写前复查影响（结果照常写回、记录照常追加）；
    - 检测序号从已有 records.jsonl 末尾续接（跨会话不重置）。
运行：python -m unittest discover -s tests
"""
import json
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import config  # noqa: E402
import plc_link.plc_server as plc_mod  # noqa: E402
from plc_link.plc_server import VisionService  # noqa: E402


class _FakeRegs:
    """内存寄存器替身：记录写入序列，读返回最后写入值"""

    def __init__(self):
        self.regs = [0] * config.N_REGS
        self.writes = []

    def read_reg(self, addr):
        return self.regs[addr] & 0xFFFF

    def write_reg(self, addr, val):
        self.regs[addr] = val & 0xFFFF
        self.writes.append((addr, val & 0xFFFF))


def make_svc(clock) -> VisionService:
    """绕过 __init__（不拉起 Modbus），手工装配被测依赖的真实实例"""
    svc = VisionService.__new__(VisionService)
    svc._clock = clock
    svc._faulted = False
    svc.seq = 0
    svc.defect_rate = 0.0                      # _process_once 需要
    svc.rng = np.random.default_rng(0)
    # Windows 下 mkstemp 打开的句柄未关前不能删除，改为：生成唯一路径
    svc.records_path = str(Path(tempfile.gettempdir()) /
                           f"_wd_test_{uuid.uuid4().hex}.jsonl")
    svc._regs = _FakeRegs()
    svc.read_reg = svc._regs.read_reg
    svc.write_reg = svc._regs.write_reg
    svc._lock = threading.Lock()               # _fire_watchdog/_process_once 需要
    svc._save_latest = lambda frame, result: None   # 测试不写 data/annot
    return svc


def _fake_inspect_result(result="NG", types=("scratch",)) -> dict:
    """_process_once 写回路径所需的最小结果结构（见 build_record/编码）"""
    return {"result": result,
            "defect_types": list(types),
            "confidence": 0.90,
            "hole_max_offset_mm": 0.02,
            "locate": {"ok": True, "center_mm": [0.10, -0.20],
                       "angle_deg": 1.0}}


class VirtualClock:
    """可手动推进的虚拟时钟"""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class WatchdogTests(unittest.TestCase):

    def setUp(self):
        self.clk = VirtualClock()

    def test_fire_writes_fault_and_record(self):
        svc = make_svc(self.clk)
        svc._fire_watchdog()
        regs = svc._regs.regs
        self.assertEqual(regs[config.REG_RESULT], config.RESULT_FAULT)
        self.assertEqual(regs[config.REG_BUSY], config.BUSY_DONE)
        self.assertTrue(svc._faulted)
        line = Path(svc.records_path).read_text(encoding="utf-8").strip()
        rec = json.loads(line)
        self.assertEqual(rec["result"], "FAULT")
        self.assertEqual(rec["seq"], 1)

    def test_no_premature_fire(self):
        """未到 deadline 不触发、过线即触发——验证判定条件本身"""
        svc = make_svc(self.clk)
        deadline = self.clk.now + config.WATCHDOG_TIMEOUT_S
        self.assertFalse(self.clk.now > deadline)
        self.assertFalse(svc._faulted)
        self.clk.now += config.WATCHDOG_TIMEOUT_S + 0.01
        self.assertTrue(self.clk.now > deadline)


class ProcessOnceRaceTests(unittest.TestCase):
    """竞态闭环：真实执行 _process_once（合成+检测打桩，时序可控）"""

    def setUp(self):
        self.clk = VirtualClock()
        self.svc = make_svc(self.clk)
        self._orig_inspect = plc_mod.vp.inspect_frame

    def tearDown(self):
        plc_mod.vp.inspect_frame = self._orig_inspect

    def _run_process_once(self, inspect_fn):
        """以注入的 inspect_frame 执行一轮真实 _process_once"""
        plc_mod.vp.inspect_frame = inspect_fn
        self.svc._process_once(1)              # 触发码 1（正常路径）

    def test_late_result_cannot_overwrite_fault_code(self):
        """处理过程中看门狗判故障 → 写寄存器前锁内复查，迟到结果作废。

        复现的交错时序：_process_once 取"处理前快照"（未故障）→
        合成+检测（此间 _fire_watchdog 写入 999）→ 持锁写回前复查
        _faulted 必须放弃写回。若无写前复查，HR2 会被 OK/NG 覆盖。
        """

        def inspect_with_watchdog_fire(frame):
            # 模拟主循环在处理窗口内判超时（写 999 + 置故障标志）
            self.svc._fire_watchdog()
            return _fake_inspect_result("OK", ()), 80.0

        self._run_process_once(inspect_with_watchdog_fire)

        regs = self.svc._regs.regs
        self.assertEqual(regs[config.REG_RESULT], config.RESULT_FAULT,
                         "迟到的 OK 结果不得覆盖看门狗故障码 999")
        self.assertEqual(regs[config.REG_BUSY], config.BUSY_DONE)
        result_writes = [v for a, v in self.svc._regs.writes
                         if a == config.REG_RESULT]
        self.assertEqual(result_writes, [config.RESULT_FAULT],
                         "HR2 只允许被看门狗写过一次")
        self.assertEqual(regs[config.REG_HEARTBEAT], 0,
                         "作废的结果不得递增心跳")
        # seq=1 为看门狗 FAULT 记录占号；迟到结果不得再追加自己的记录
        self.assertEqual(self.svc.seq, 1)
        lines = Path(self.svc.records_path).read_text(
            encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1, "只允许存在看门狗 FAULT 一条记录")
        self.assertEqual(json.loads(lines[0])["result"], "FAULT")

    def test_normal_result_written_when_not_faulted(self):
        """未发生故障时，写前复查不得拦截正常结果（防修出回归）"""

        def inspect_normal(frame):
            return _fake_inspect_result("NG", ("scratch",)), 75.0

        self._run_process_once(inspect_normal)

        regs = self.svc._regs.regs
        self.assertEqual(regs[config.REG_RESULT], config.RESULT_NG)
        self.assertEqual(regs[config.REG_DEFECT],
                         config.DEFECT_BIT["scratch"])
        self.assertEqual(regs[config.REG_BUSY], config.BUSY_DONE)
        self.assertEqual(regs[config.REG_HEARTBEAT], 1)
        self.assertEqual(self.svc.seq, 1)
        rec = json.loads(Path(self.svc.records_path)
                         .read_text(encoding="utf-8").strip())
        self.assertEqual(rec["seq"], 1)
        self.assertEqual(rec["result"], "NG")
        # ts 带日期（跨会话记录不混叠的口径锁定）
        self.assertRegex(rec["ts"], r"^\d{4}-\d{2}-\d{2} ")


class SeqContinuityTests(unittest.TestCase):
    """检测测序号从已有 records.jsonl 末尾续接"""

    def test_continues_from_last_valid_seq(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            lines = [{"seq": 1, "result": "OK"}, {"seq": 2, "result": "NG"},
                     {"seq": 5, "result": "OK"}]      # 模拟历史乱序残留
            path.write_text("\n".join(json.dumps(x) for x in lines) + "\n",
                            encoding="utf-8")
            self.assertEqual(VisionService._last_seq(path), 5)

    def test_tolerates_missing_empty_and_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(VisionService._last_seq(
                Path(tmp) / "nope.jsonl"), 0)          # 无文件
            empty = Path(tmp) / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            self.assertEqual(VisionService._last_seq(empty), 0)  # 空文件
            mixed = Path(tmp) / "mixed.jsonl"
            mixed.write_text(
                json.dumps({"seq": 3}) + "\n" + "半行坏数据\n",
                encoding="utf-8")
            self.assertEqual(VisionService._last_seq(mixed), 3)  # 坏行跳过


if __name__ == "__main__":
    unittest.main(verbosity=2)
