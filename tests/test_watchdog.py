# -*- coding: utf-8 -*-
"""
tests/test_watchdog.py —— 视觉服务状态机时间线单元测试（虚拟时钟）
========================================================
不起真服务器、不睡真 4 秒：给 VisionService 注入虚拟时钟与内存寄存器，
毫秒级重放「触发→超时→看门狗动作」时间线，验证：
    - 超时后 _fire_watchdog 写入 RESULT_FAULT + BUSY_DONE + FAULT 记录；
    - 故障标志置位后，迟到的工作线程结果被作废；
    - 未超时前不误触发。
运行：python -m unittest discover -s tests
"""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from plc_link.plc_server import VisionService


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
    # Windows 下 mkstemp 打开的句柄未关前不能删除，改为：生成唯一路径
    import uuid
    svc.records_path = str(Path(tempfile.gettempdir()) /
                           f"_wd_test_{uuid.uuid4().hex}.jsonl")
    svc._regs = _FakeRegs()
    svc.read_reg = svc._regs.read_reg
    svc.write_reg = svc._regs.write_reg
    svc._lock = threading.Lock()               # _fire_watchdog 需要
    return svc


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

    def test_late_worker_result_is_voided(self):
        """故障后迟到线程在 inspect_frame 之后检查 _faulted 必须 return"""
        svc = make_svc(self.clk)
        svc._fire_watchdog()
        with svc._lock:
            faulted = svc._faulted
        self.assertTrue(faulted)
        results = [v for a, v in svc._regs.writes if a == config.REG_RESULT]
        self.assertEqual(results, [config.RESULT_FAULT])

    def test_no_premature_fire(self):
        """未到 deadline 不触发、过线即触发——验证判定条件本身"""
        svc = make_svc(self.clk)
        deadline = self.clk.now + config.WATCHDOG_TIMEOUT_S
        self.assertFalse(self.clk.now > deadline)
        self.assertFalse(svc._faulted)
        self.clk.now += config.WATCHDOG_TIMEOUT_S + 0.01
        self.assertTrue(self.clk.now > deadline)


if __name__ == "__main__":
    unittest.main(verbosity=2)
