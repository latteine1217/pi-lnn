# docs/superpowers/ — 設計流程歷史紀錄

本資料夾保存專案各階段「Plan → Spec」的設計決策歷史，採用 brainstorm-then-specify 工作流的產出。

## 角色

- `plans/`: 高層需求與決策路徑（What / Why）。每份對應一次重大方向變更。
- `specs/`: 落地實作規格（How，含介面、loss、資料流、檢核點）。

## ⚠️ 仍被引用，請勿移動

specs/ 內多份文件**目前仍被 src 與 configs 引用**，移動或重命名會破壞參照：

| Spec | 被引用處 |
|---|---|
| `specs/2026-04-26-pi-lnn-package-refactor-design.md` | `src/pi_lnn/training.py` |
| `specs/2026-05-04-al-continuity-design.md` | `src/pi_lnn/config.py`、`src/pi_lnn/losses.py`、`configs/exp_070~072` |

ADR-001 §123 亦明定「個別實驗的詳細 spec 放於本資料夾」。

## 與 experiment_log.md 的分工

- 本資料夾：**設計階段**的 plan / spec（決策過程、權衡、預期）。
- `docs/experiment_log.md`：實驗 state 主檔（STATE/INDEX 結論層）。詳細 RECORD 已拆檔（2026-05-15）：
  - EXP-001 ~ EXP-063 → `docs/experiment_archive_kolmogorov.md`
  - EXP-064 ~ EXP-101 → `docs/experiment_archive_kolmogorov_post_k100.md`
  - Cylinder CEXP → `docs/cylinder_log.md`

新增 spec 時請沿用既有命名：`YYYY-MM-DD-<topic>-design.md`。
