# -*- coding: utf-8 -*-
"""
plc_link/modbus_client_test.py —— Modbus 客户端手动读写自测脚本
================================================================
职责：
    以"上位机/PLC 客户端"的身份对 plc_server.py 做端到端联调自测：
      1) 连接从站，读取 HR0~HR10 初始状态；
      2) 写 HR0=1 触发一次完整检测，轮询 HR1 直到完成，读回并解码
         HR2~HR6、HR10（含 0.01mm/0.1° 定点数与缺陷位解码）；
      3) 连续触发多次，统计结果分布与心跳；
      4) 看门狗验证：写 HR0=2（触发并模拟处理卡死）→ 等待完成 →
         校验 HR2==999（故障码）；
      5) 心跳验证：空闲 2.2 秒，校验 HR10 递增。

用法（先在另一个终端启动服务端）：
    终端A: python plc_link/plc_server.py
    终端B: python plc_link/modbus_client_test.py
可选参数：--host 127.0.0.1 --port 502 --triggers 3 --skip-watchdog

退出码：全部通过=0，任一失败=1（可直接用于自动化验收）。
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# 允许直接 `python plc_link/modbus_client_test.py` 运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from pymodbus.client import ModbusTcpClient

logging.getLogger("pymodbus").setLevel(logging.WARNING)

PASS, FAIL = "PASS", "FAIL"
_results = []                      # [(项目名, PASS/FAIL, 说明)]


def report(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" —— {detail}" if detail
                                                  else ""))


def decode_defect_code(code: int) -> list:
    """把 HR3 缺陷类型码解码为类型名列表"""
    names = []
    for name, bit in config.DEFECT_BIT.items():
        if code & bit:
            names.append(name)
    return names


def read_regs(client: ModbusTcpClient, addr: int, count: int) -> list:
    """读保持寄存器（带错误检查）"""
    rr = client.read_holding_registers(addr, count)
    if rr.isError():
        raise IOError(f"读寄存器 HR{addr}×{count} 失败: {rr}")
    return list(rr.registers)


def read_reg(client: ModbusTcpClient, addr: int) -> int:
    return read_regs(client, addr, 1)[0]


def write_reg(client: ModbusTcpClient, addr: int, value: int) -> None:
    rr = client.write_register(addr, value)
    if rr.isError():
        raise IOError(f"写寄存器 HR{addr}={value} 失败: {rr}")


def wait_done(client: ModbusTcpClient, timeout_s: float = 6.0) -> int:
    """轮询 HR1 直到 BUSY_DONE(2) 或超时；返回最后读到的 HR1"""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if read_reg(client, config.REG_BUSY) == config.BUSY_DONE:
            return config.BUSY_DONE
        time.sleep(0.02)
    return read_reg(client, config.REG_BUSY)


def print_state(client: ModbusTcpClient, tag: str) -> None:
    """打印 HR0~HR10 的解码视图"""
    regs = read_regs(client, 0, 11)
    print(f"  [{tag}] HR0触发={regs[0]} HR1状态={regs[1]} "
          f"HR2结果={regs[2]} HR3缺陷=0b{regs[3]:07b}"
          f"({','.join(decode_defect_code(regs[3])) or '无'}) "
          f"HR4={config.from_int16(regs[4])}(0.01mm) "
          f"HR5={config.from_int16(regs[5])}(0.01mm) "
          f"HR6={config.from_int16(regs[6])}(0.1°) HR10心跳={regs[10]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Modbus 客户端联调自测（需先启动 plc_server.py）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--host", default=config.MODBUS_HOST, help="服务端地址")
    ap.add_argument("--port", type=int, default=config.MODBUS_PORT,
                    help="服务端端口")
    ap.add_argument("--triggers", type=int, default=3,
                    help="正常触发次数")
    ap.add_argument("--skip-watchdog", action="store_true",
                    help="跳过看门狗测试（不想等 4 秒时使用）")
    args = ap.parse_args()

    client = ModbusTcpClient(args.host, port=args.port)
    print(f"连接 Modbus/TCP 从站 {args.host}:{args.port} ...")
    if not client.connect():
        print(f"[FAIL] 无法连接——请先在另一个终端运行 "
              f"python plc_link/plc_server.py")
        sys.exit(1)
    report("连接从站", True, f"{args.host}:{args.port}")

    try:
        # ---------- 1) 初始状态 ----------
        print("\n===== 1/5 读取初始寄存器状态 =====")
        print_state(client, "初始")
        report("初始状态合理", read_reg(client, config.REG_BUSY) in
               (config.BUSY_IDLE, config.BUSY_DONE))

        # ---------- 2) 正常触发一次 ----------
        print("\n===== 2/5 写 HR0=1 触发一次完整检测 =====")
        write_reg(client, config.REG_TRIGGER, 1)
        state = wait_done(client)
        ok = state == config.BUSY_DONE
        report("触发→完成（HR1=2）", ok,
               f"HR1={state}（正常检测约 0.2s，看门狗 2s）")
        print_state(client, "检测后")
        result_code = read_reg(client, config.REG_RESULT)
        report("结果码有效", result_code in (config.RESULT_OK,
                                             config.RESULT_NG),
               f"HR2={result_code}"
               f"({'OK' if result_code == 1 else 'NG'})")
        dev_x_mm = config.from_int16(read_reg(client, config.REG_DEV_X)) / 100
        dev_y_mm = config.from_int16(read_reg(client, config.REG_DEV_Y)) / 100
        angle = config.from_int16(read_reg(client, config.REG_ANGLE)) / 10
        report("定位偏差在合理范围（±10mm）",
               abs(dev_x_mm) <= 10.0 and abs(dev_y_mm) <= 10.0,
               f"X={dev_x_mm:+.2f}mm Y={dev_y_mm:+.2f}mm θ={angle:+.1f}°")

        # ---------- 3) 连续触发统计 ----------
        print(f"\n===== 3/5 连续触发 {args.triggers} 次 =====")
        ng = ok_cnt = 0
        hb_before = read_reg(client, config.REG_HEARTBEAT)
        for i in range(args.triggers):
            write_reg(client, config.REG_TRIGGER, 1)
            st = wait_done(client)
            rc = read_reg(client, config.REG_RESULT)
            if st != config.BUSY_DONE or rc not in (1, 2):
                report(f"第{i + 1}次触发", False, f"HR1={st} HR2={rc}")
            ng += (rc == config.RESULT_NG)
            ok_cnt += (rc == config.RESULT_OK)
            types = decode_defect_code(read_reg(client, config.REG_DEFECT))
            print(f"  第{i + 1}次: HR2={rc}({ 'NG' if rc == 2 else 'OK'}) "
                  f"缺陷={types or '无'} "
                  f"心跳={read_reg(client, config.REG_HEARTBEAT)}")
        hb_after = read_reg(client, config.REG_HEARTBEAT)
        report("多次触发全部完成",
               ok_cnt + ng == args.triggers,
               f"OK×{ok_cnt} NG×{ng}（缺陷率见服务端参数）")
        report("心跳随检测递增", hb_after > hb_before,
               f"HR10 {hb_before} → {hb_after}")

        # ---------- 4) 看门狗验证 ----------
        if args.skip_watchdog:
            print("\n===== 4/5 看门狗验证（已跳过） =====")
        else:
            print("\n===== 4/5 看门狗验证：写 HR0=2 模拟处理卡死 =====")
            write_reg(client, config.REG_TRIGGER, 2)
            # 等到超过看门狗时限（触发 2s 看门狗 + 模拟卡死 3.5s 释放线程）
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < config.WATCHDOG_TIMEOUT_S + 3.0:
                if read_reg(client, config.REG_BUSY) == config.BUSY_DONE:
                    break
                time.sleep(0.05)
            fault_code = read_reg(client, config.REG_RESULT)
            report("看门狗写故障码 999",
                   fault_code == config.RESULT_FAULT,
                   f"HR2={fault_code}（等待 "
                   f"{time.perf_counter() - t0:.1f}s）")
            # 看门狗触发后应能继续正常工作（迟到的结果未覆盖故障码）
            write_reg(client, config.REG_TRIGGER, 1)
            st = wait_done(client)
            rc = read_reg(client, config.REG_RESULT)
            report("故障后恢复正常触发",
                   st == config.BUSY_DONE and rc in (1, 2),
                   f"HR1={st} HR2={rc}")

        # ---------- 5) 空闲心跳 ----------
        print("\n===== 5/5 空闲心跳验证（等待 2.2s）=====")
        hb1 = read_reg(client, config.REG_HEARTBEAT)
        time.sleep(2.2)
        hb2 = read_reg(client, config.REG_HEARTBEAT)
        report("空闲心跳按周期递增", (hb2 - hb1) & 0xFFFF >= 1,
               f"HR10 {hb1} → {hb2}")

    finally:
        client.close()

    # ---------- 汇总 ----------
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    print("\n===== 自测汇总 =====")
    for name, st, detail in _results:
        print(f"  [{st}] {name}" + (f" —— {detail}" if detail else ""))
    print(f"共 {len(_results)} 项，通过 {len(_results) - n_fail}，"
          f"失败 {n_fail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
