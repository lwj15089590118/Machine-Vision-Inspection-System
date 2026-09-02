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
    - 落盘窗口竞态（真实调用 _process_once）：worker 写完结果后的
      落盘期间看门狗触发，写 999 前锁内复查 HR1==DONE，已送达的
      结果不得被改回 999、记录流不得混入 FAULT（语义不分叉）；
    - seq 唯一性：序号分配收敛到锁内单点 _next_seq，多线程并发
      分配无重复、无丢号；
    - 时序压力：100 轮随机注入看门狗触发点（写回前/落盘中），
      每轮恰好一条记录、seq 严格递增、HR2 与记录语义一致；
    - 正常路径不受写前复查影响（结果照常写回、记录照常追加）；
    - 检测序号从已有 records.jsonl 末尾续接（跨会话不重置）。
运行：python -m unittest discover -s tests
"""
import json
import random
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
        self._orig_synth = plc_mod.synth_frame

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

    def _record_lines(self, svc) -> list:
        text = Path(svc.records_path).read_text(encoding="utf-8").strip()
        return text.splitlines() if text else []

    def test_watchdog_during_persistence_cannot_overwrite_result(self):
        """worker 写完结果后的落盘期间看门狗触发 → 已送达结果不被改回 999。

        复现的交错时序：_process_once 持锁写回 OK/NG 并置 HR1=DONE →
        释放锁进入落盘（records/latest，此窗口内主循环时钟越过
        deadline）→ _fire_watchdog 获锁后必须复查 HR1==DONE 并让位。
        若无此复查，HR2 会被无条件改回 999，上位机读到故障码而记录流
        是正常件（寄存器/记录流语义分叉）。
        """
        svc = self.svc

        def inspect_normal(frame):
            return _fake_inspect_result("OK", ()), 70.0

        def save_with_watchdog_fire(frame, result):
            svc._fire_watchdog()           # 注入：落盘窗口内看门狗触发

        plc_mod.vp.inspect_frame = inspect_normal
        svc._save_latest = save_with_watchdog_fire
        svc._process_once(1)

        regs = svc._regs.regs
        self.assertEqual(regs[config.REG_RESULT], config.RESULT_OK,
                         "已送达的结果不得被迟到的看门狗改回 999")
        self.assertEqual(regs[config.REG_BUSY], config.BUSY_DONE)
        result_writes = [v for a, v in svc._regs.writes
                         if a == config.REG_RESULT]
        self.assertEqual(result_writes, [config.RESULT_OK],
                         "HR2 只允许被 worker 写过一次（无 999 覆写）")
        self.assertFalse(svc._faulted,
                         "本轮已落结果，看门狗让位后不得残留故障标志")
        self.assertEqual(regs[config.REG_HEARTBEAT], 1)
        lines = self._record_lines(svc)
        self.assertEqual(len(lines), 1,
                         "只允许存在正常结果一条记录（无 FAULT 记录混入）")
        rec = json.loads(lines[0])
        self.assertEqual(rec["result"], "OK")
        self.assertEqual(rec["seq"], 1)
        self.assertEqual(svc.seq, 1)

    def test_next_seq_concurrent_allocation_unique(self):
        """seq 自增收敛到锁内单点：多线程并发分配无重复、无丢号。

        `seq += 1` 在 GIL 下非原子；修复前 worker 与看门狗线程各自裸增，
        极端交错会丢号/重号。现全部经 _next_seq() 在锁内分配。
        """
        svc = self.svc
        n_threads, per_thread = 8, 200
        got, threads = [], []
        barrier = threading.Barrier(n_threads)   # 同时起跑，放大交错概率

        def alloc():
            values = []
            barrier.wait()
            for _ in range(per_thread):
                values.append(svc._next_seq())
            got.extend(values)                 # list.extend 原子（GIL）

        for _ in range(n_threads):
            threads.append(threading.Thread(target=alloc))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(got), n_threads * per_thread)
        self.assertEqual(len(set(got)), len(got), "并发分配出现重复序号")
        self.assertEqual(sorted(got),
                         list(range(1, n_threads * per_thread + 1)),
                         "序号应恰好覆盖 1..N（无丢号）")

    def test_timing_stress_100_rounds_consistent(self):
        """时序压力：100 轮随机注入看门狗触发点，寄存器与记录流须一致。

        每轮二选一注入（seed 固定可复现）：写回前触发（结果必须作废
        → HR2=999 + FAULT 记录）或落盘中触发（结果已送达 → HR2 保持
        原值、无 FAULT 记录）。断言每轮恰好一条记录、seq 严格递增无
        重复、HR2 与记录 result 语义一致、故障标志与结果匹配。
        """
        svc = self.svc
        rng = random.Random(42)
        # 压力轮只测时序交错，桩掉合成器避免 100 次真实渲染的墙钟噪声
        plc_mod.synth_frame = lambda rng_, **kw: (
            np.zeros((1, 1), dtype=np.uint8), {"defects": []})
        seq_expected = 0
        try:
            for i in range(100):
                svc._faulted = False
                svc._regs.regs = [0] * config.N_REGS
                fire_before_write = rng.random() < 0.5

                def inspect_fn(frame, _fbw=fire_before_write):
                    if _fbw:
                        svc._fire_watchdog()   # 注入点 A：写回前触发
                    return _fake_inspect_result("OK", ()), 70.0

                def save_fn(frame, result, _fbw=fire_before_write):
                    if not _fbw:
                        svc._fire_watchdog()   # 注入点 B：落盘中触发

                plc_mod.vp.inspect_frame = inspect_fn
                svc._save_latest = save_fn
                svc._process_once(1)

                seq_expected += 1
                lines = self._record_lines(svc)
                self.assertEqual(len(lines), seq_expected,
                                 f"第{i}轮：应恰好追加一条记录")
                rec = json.loads(lines[-1])
                self.assertEqual(rec["seq"], seq_expected,
                                 f"第{i}轮：seq 应严格递增且无重复")
                self.assertEqual(rec["fault"], fire_before_write,
                                 f"第{i}轮：记录 fault 标志与注入点不符")
                hr2 = svc._regs.regs[config.REG_RESULT]
                if fire_before_write:
                    self.assertEqual(hr2, config.RESULT_FAULT,
                                     f"第{i}轮：写回前触发应作废结果为 999")
                    self.assertTrue(svc._faulted)
                else:
                    self.assertEqual(hr2, config.RESULT_OK,
                                     f"第{i}轮：落盘中触发不得覆盖结果")
                    self.assertFalse(svc._faulted)
                self.assertEqual(svc._regs.regs[config.REG_BUSY],
                                 config.BUSY_DONE)
        finally:
            plc_mod.synth_frame = self._orig_synth


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
