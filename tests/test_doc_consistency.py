# -*- coding: utf-8 -*-
"""
tests/test_doc_consistency.py —— 工件↔文档数字一致性自检（unittest 薄壳）
========================================================
三层防线（引擎在 tools/doc_consistency.py，零依赖）：
  A. RepoGateTests    —— 对仓库当前工件与三份文档实跑全量校验，必须全绿。
     第一轮 P0（README 宣称 500 帧而工件 100 帧）即数字口径漂移；重跑
     run_batch 后若忘记同步 README/测试报告/设计说明书，本类必红，
     封死"手动同步快照"（复审 N3）的复发通道。
  B. TamperDetectionTests —— 把工件与文档复制到临时目录后"故意篡改"，
     验证校验器真能抓到漂移（防"校验器自身失效却永远绿灯"的元风险）。
     只动临时副本，绝不修改仓库真文件。
  C. CheckerToleranceTests —— 数字格式容错语义：千分位/全角负号/单位/
     舍入半刻度/整数零容差/缺锚点报错/相对容差须显式声明。
  D. CliTests —— 命令行退出码（0=一致 / 1=漂移），可直接挂 CI。
运行：python -m unittest discover -s tests
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import doc_consistency as dc

ROOT = dc.PROJECT_ROOT


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ================================================================
# A. 仓库现状门禁（防漂移复发的常驻闸门）
# ================================================================
class RepoGateTests(unittest.TestCase):

    def test_artifact_and_all_docs_are_consistent(self):
        failures, summary = dc.check(ROOT)
        self.assertEqual(
            failures, [],
            "工件↔文档数字一致性被破坏（重跑批量后未同步文档？）：\n"
            + "\n".join(failures))
        self.assertGreater(summary["claims"], 100,
                           f"声称清单异常缩水：{summary}")
        self.assertGreater(summary["matches"], 100,
                           f"逐处比对数异常：{summary}")

    def test_claim_inventory_covers_key_fields(self):
        """防"校验清单被悄悄掏空"：关键字段必须有三份文档的声称覆盖"""
        artifact = json.loads(
            (ROOT / dc.ARTIFACT_RELPATH).read_text(encoding="utf-8"))
        by_doc = {}
        for c in dc.build_claims(artifact):
            by_doc.setdefault(c.doc, set()).add(c.field)

        types = {r["type"] for r in artifact["type_rows"] if r["injected"]}
        base = ({"meta.count", "meta.defect_rate", "meta.seed",
                 "confusion.tp", "confusion.fn", "confusion.fp",
                 "confusion.tn", "recall_piece", "false_alarm", "precision",
                 "locate.center_err_mm.p95", "locate.angle_err_deg.p95",
                 "takt_ms.mean", "takt_ms.p95"}
                | {f"type_rows[{t}].recall" for t in types})
        for doc in dc.DOC_RELPATHS:
            self.assertEqual(base - by_doc.get(doc, set()), set(),
                             f"{doc} 缺少关键字段的一致性声称")

        report_only = ({"meta.generated_at", "meta.wall_time_s",
                        "meta.env.os", "confusion.ng_total",
                        "confusion.ok_total", "confusion.locate_fail",
                        "takt_ms.max",
                        "gates[name=件级缺陷检出率].actual"}
                       | {f"locate.{k}.{s}"
                          for k in ("center_err_px", "center_err_mm",
                                    "angle_err_deg", "scale_err_pct")
                          for s in ("mean", "std", "p95", "max")})
        self.assertEqual(
            report_only - by_doc.get("docs/测试报告.md", set()), set(),
            "docs/测试报告.md 缺少与 batch_report.json 同源校验的声称")


# ================================================================
# B. 篡改注入（临时副本，验证校验器能抓到真实漂移）
# ================================================================
class TamperDetectionTests(unittest.TestCase):

    def setUp(self):
        tmp = Path(tempfile.mkdtemp(prefix="doc_consistency_tamper_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        self.tmp = tmp
        for rel in (dc.ARTIFACT_RELPATH, *dc.DOC_RELPATHS):
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dst)

    def _rewrite(self, rel: str, text: str) -> None:
        (self.tmp / rel).write_text(text, encoding="utf-8")

    def _tamper(self, rel: str, old: str, new: str, count: int = 1) -> None:
        text = (self.tmp / rel).read_text(encoding="utf-8")
        self.assertIn(old, text, f"篡改前提不成立：{rel} 中找不到 {old!r}")
        self._rewrite(rel, text.replace(old, new, count))

    def test_pristine_copy_is_green(self):
        failures, _ = dc.check(self.tmp)
        self.assertEqual(failures, [], "未篡改的副本不应报漂移：\n"
                         + "\n".join(failures))

    def test_tampered_artifact_count_is_caught(self):
        """第一轮 P0 场景重演：工件帧数变了而文档没跟上"""
        path = self.tmp / dc.ARTIFACT_RELPATH
        art = json.loads(path.read_text(encoding="utf-8"))
        art["meta"]["count"] = 100
        path.write_text(json.dumps(art, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        failures, _ = dc.check(self.tmp)
        self.assertTrue(any("meta.count" in f and "README.md" in f
                            for f in failures),
                        "应指出 README 的帧数声称与工件 meta.count 漂移：\n"
                        + "\n".join(failures))
        self.assertTrue(any("docs/系统设计说明书.md" in f for f in failures),
                        "设计说明书的帧数声称也应被点名")

    def test_snapshot_not_re_synced_is_caught(self):
        """复审 N3 场景：重跑后测试报告已更新、docs/batch_report.json 忘了 cp"""
        self._tamper(dc.ARTIFACT_RELPATH, '"generated_at": "',
                     '"generated_at": "2099-')  # 快照时间戳与报告不再同源
        failures, _ = dc.check(self.tmp)
        self.assertTrue(any("meta.generated_at" in f
                            and "docs/测试报告.md" in f for f in failures),
                        "快照与测试报告不同源必须被抓到：\n"
                        + "\n".join(failures))

    def test_tampered_readme_tp_is_caught(self):
        self._tamper("README.md", "TP=241", "TP=240")
        failures, _ = dc.check(self.tmp)
        hit = [f for f in failures if "confusion.tp" in f]
        self.assertTrue(hit, "应指出 README 的 TP 声称漂移：\n"
                        + "\n".join(failures))
        self.assertIn("README.md", hit[0])
        self.assertIn("240", hit[0])
        self.assertIn("241", hit[0])

    def test_tampered_report_takt_is_caught(self):
        self._tamper("docs/测试报告.md", "**77.6 ms**", "**66.6 ms**")
        failures, _ = dc.check(self.tmp)
        self.assertTrue(any("takt_ms.mean" in f and "docs/测试报告.md" in f
                            for f in failures),
                        "应指出测试报告节拍与工件 takt_ms.mean 漂移：\n"
                        + "\n".join(failures))

    def test_tampered_design_type_recall_is_caught(self):
        self._tamper("docs/系统设计说明书.md", "78.9%", "80.9%")
        failures, _ = dc.check(self.tmp)
        hit = [f for f in failures if "type_rows[stain].recall" in f]
        self.assertTrue(hit, "应指出设计说明书污渍检出率漂移：\n"
                        + "\n".join(failures))
        self.assertIn("docs/系统设计说明书.md", hit[0])
        self.assertIn("0.7887", hit[0])   # 消息须给出工件字段值与差值


# ================================================================
# C. 容错语义（合成数据，不依赖仓库文件内容）
# ================================================================
def _mini_artifact():
    return {"confusion": {"tp": 241, "fn": 0},
            "locate": {"angle_err_deg": {"mean": -0.0196}},
            "takt_ms": {"mean": 77.6},
            "recall_piece": 0.8667,
            "meta": {"env": {"python": "3.12.10"}}}


def _run(claims, text, artifact=None):
    failures, _ = dc.check_texts(artifact or _mini_artifact(),
                                 {"t.md": text}, claims)
    return failures


class CheckerToleranceTests(unittest.TestCase):

    TP = dc.Claim("t.md", "confusion.tp", r"TP\s*=\s*(\d[\d,]*)")
    ANGLE = dc.Claim("t.md", "locate.angle_err_deg.mean",
                     r"均值\s*(" + dc.NUM_SIGNED + r")°")
    RECALL = dc.Claim("t.md", "recall_piece", r"([\d.,]+)\s*%", scale=0.01)
    TAKT = dc.Claim("t.md", "takt_ms.mean", r"平均\s*([\d.,]+)\s*ms")
    PYENV = dc.Claim("t.md", "meta.env.python", r"Python\s*([\d.]+)",
                     1.0, "text")

    def test_thousand_separator_and_fullwidth_sign_tolerated(self):
        text = "TP = 1,241\n均值 −0.020°\n平均 77.60 ms"
        self.assertEqual(_run([self.TP, self.ANGLE, self.TAKT], text,
                              {"confusion": {"tp": 1241},
                               "locate": {"angle_err_deg": {"mean": -0.0196}},
                               "takt_ms": {"mean": 77.6}}), [])

    def test_rounding_half_ulp_tolerated_but_real_drift_fails(self):
        self.assertEqual(_run([self.RECALL], "86.7%"), [])      # 0.867≈0.8667
        self.assertEqual(len(_run([self.RECALL], "86.6%")), 1)  # 0.866 差 7e-4
        self.assertEqual(len(_run([self.RECALL], "86.8%")), 1)  # 0.868 差 13e-4

    def test_integer_fields_have_zero_tolerance(self):
        fails = _run([self.TP], "TP = 240")     # 差 1 也必须红
        self.assertEqual(len(fails), 1)
        self.assertIn("confusion.tp", fails[0])
        self.assertIn("241", fails[0])

    def test_missing_anchor_fails_loudly(self):
        fails = _run([self.TP], "这一段没有任何混淆矩阵数字")
        self.assertEqual(len(fails), 1)
        self.assertIn("锚点缺失", fails[0])
        self.assertIn("confusion.tp", fails[0])

    def test_text_claim_mismatch_is_reported(self):
        self.assertEqual(_run([self.PYENV], "Python 3.12.10 · OpenCV"), [])
        fails = _run([self.PYENV], "Python 3.11.9 · OpenCV")
        self.assertEqual(len(fails), 1)
        self.assertIn("3.11.9", fails[0])
        self.assertIn("3.12.10", fails[0])

    def test_rel_tol_is_opt_in_and_must_be_documented(self):
        """默认 0 容差抓真漂移；±5% 相对容差仅对显式声明 note 的条目生效"""
        self.assertEqual(len(_run([self.TAKT], "平均 79.5 ms")), 1)
        loose = dc.Claim("t.md", "takt_ms.mean", r"平均\s*([\d.,]+)\s*ms",
                         rel_tol=0.05, note="墙钟演示专用")
        self.assertEqual(_run([loose], "平均 79.5 ms"), [])

    def test_drift_message_points_to_file_line_and_field(self):
        art = _mini_artifact()
        text = "TP=241 没问题\nTP = 200 有问题"
        fails, _ = dc.check_texts(art, {"t.md": text},
                                  [dc.Claim("t.md", "confusion.tp",
                                            r"TP\s*=\s*(\d[\d,]*)")])
        self.assertEqual(len(fails), 1)
        self.assertIn("t.md:2", fails[0])           # 行号可定位
        self.assertIn("TP = 200", fails[0])         # 引用原句
        self.assertIn("confusion.tp = 241", fails[0])  # 工件字段与值
        self.assertIn("-41", fails[0])              # 差多少


# ================================================================
# D. CLI 退出码（可挂 CI）
# ================================================================
class CliTests(unittest.TestCase):

    def _run_cli(self, *args):
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "doc_consistency.py"),
             *args],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=env, timeout=60)

    def test_cli_green_on_repo(self):
        p = self._run_cli()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("DOC-ARTIFACT CONSISTENT", p.stdout)

    def test_cli_exit_nonzero_on_drift(self):
        tmp = Path(tempfile.mkdtemp(prefix="doc_consistency_cli_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "docs").mkdir()
        for rel in (dc.ARTIFACT_RELPATH, *dc.DOC_RELPATHS):
            shutil.copy2(ROOT / rel, tmp / rel)
        text = (tmp / "README.md").read_text(encoding="utf-8")
        (tmp / "README.md").write_text(text.replace("TP=241", "TP=99"),
                                       encoding="utf-8")
        p = self._run_cli("--root", str(tmp))
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("DOC-ARTIFACT DRIFT", p.stdout)
        self.assertIn("confusion.tp", p.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
