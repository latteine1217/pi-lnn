# AL-Continuity Infrastructure & EXP-070..072 — Design Spec

**Date:** 2026-05-04
**Scope:** Phase 2 第一波實驗 — Augmented Lagrangian on continuity，附 Pressure Poisson 對照組。
**Status:** Draft, awaiting owner approval before implementation.
**ADR reference:** [ADR-001 §4–5](../adr/001-research-priorities-and-scope.md)

---

## 1. Goal

讓 continuity constraint 從 GradNorm 的 adaptive balancing 升格成正規 constrained optimization：

$$
\mathcal{L} = w_d L_d + w_u L_{\text{ns},u} + w_v L_{\text{ns},v} + \underbrace{\big(\lambda_c C + \tfrac{\rho}{2} C^2\big)}_{\text{AL term}}
\quad\text{where}\quad C = \mathbb{E}[(\nabla\cdot u)^2]
$$

EXP-070（pure AL）採此形式；EXP-071（AL + GradNorm）將 AL term 升格為 GradNorm 第 4 task，外層多一個 $w_{\text{al}}$（見 §5）。

主要目標：把 EXP-064 的 div L2 ≈ 0.184 降到 **< 0.05**。

### Non-goals（這份 spec 不處理）

- 多 constraint AL（只處理 continuity scalar，不做 per-point Lagrangian field）
- AL 在 NG path 的整合（NG 工作在 residual vector，與 AL 的 quadratic penalty 不相容；EXP-070..071 限定 first-order optimizer）
- 改 cylinder 線（cylinder 的 BC 已用 hard output gate，不適用 AL-cont）
- Pressure Poisson 的實作（EXP-072 只是改 config，code 已有）

---

## 2. Math: Penalty Schedule with Accumulated Multiplier

### Framing 澄清（vs textbook AL）

教科書 Augmented Lagrangian 處理 signed equality constraint $g(x)=0$：

$$
\mathcal{L}_{\text{AL}} = f(x) + \lambda \cdot g(x) + \frac{\rho}{2} g(x)^2,\quad
\lambda \leftarrow \lambda + \rho \cdot g(x)
$$

其中 $g$ 可正可負，$\lambda$ 在 saddle point 收斂到 $\lambda^* = -\rho \cdot g(x^*)$，於 $g=0$ 處停止更新。

**本實作不是上述形式**。我們的 constraint 是：

$$
C = \frac{1}{N}\sum_{i=1}^{N} (\nabla\cdot u_\theta)(x_i)^2 \;\geq\; 0
$$

採 squared form 是因為 periodic Kolmogorov domain 內 $\mathrm{mean}(\nabla\cdot u) \equiv 0$（散度定理），對 signed mean 做 AL 會得到 vacuous constraint。代價是：**$C$ 永遠非負，$\lambda$ 單調非減直到 hit clip**。

因此本機制更精確的描述是：

> **Accumulated-multiplier penalty schedule** — λ 從 0 起步，隨訓練累積成「動態增長的 penalty 係數」；達 $\Lambda$ 後系統退化為 fixed quadratic penalty with coefficient $(\rho + \Lambda/C̃)$。

效果上比手動設 fixed `continuity_weight` 好：
- 早期 λ 小 → 不擾動 data fit；
- 隨 C 持續非零 → λ 自動加重；
- 一旦 C → 0 → λ 凍結，避免無止盡推高。

但**不要期望 λ 收斂到「最佳 Lagrange multiplier」**——這個機制不提供該保證。命名沿用 `AugmentedLagrangianMultiplier` 是為對齊 reviewer 與文獻常用詞，但概念上是 dynamic penalty schedule。

### Update rule

每 `al_update_freq` 步：

$$
\lambda_c \leftarrow \mathrm{clip}\big(\lambda_c + \rho \cdot \tilde{C},\; 0,\; \Lambda\big)
$$

clip lower bound = **0**（不是 $-\Lambda$）— 因為 $C \geq 0$，$\lambda < 0$ 會讓 penalty 反向，物理上無意義。

其中 $\tilde{C}$ 可選：
- `al_ema_momentum = 0`：直接用 batch constraint value（fast adaptation, noisy）
- `al_ema_momentum > 0`：EMA 平滑（`new = m·old + (1-m)·C_batch`，stable）

預設 `al_ema_momentum = 0.5`，與 GradNorm EMA 一致。

### Constraint 形式

採 **mean of squared divergence**：

$$
C = \frac{1}{N}\sum_{i=1}^{N} (\nabla\cdot u_\theta)(x_i)^2
$$

不採 $\mathrm{mean}(\nabla\cdot u)$（在 periodic domain 為 vacuous；非 periodic 也會 batch 內正負抵消）。

**注意：當 `use_sensor_physics=true`（EXP-064 預設）**，現有 `l_cont_total` 是 random-collocation `mean(cont²)` **加上** sensor 位置 `mean(cont_sp²)` 的**和**，不是單一 mean。spec §4 對此明確處理：AL 階段強制 `use_sensor_physics=false`（單一定義），或定義 $C = \tfrac{1}{2}(C_{\text{rand}} + C_{\text{sensor}})$。v1 採前者，避免雙重 mean 污染 EMA。

### λ 初值與 clip

- `al_init_lambda = 0.0`：從 unconstrained 起步，λ 隨 C 單調非減
- `al_lambda_clip = 10.0`：上限對應有效 penalty ≤ $\Lambda + \rho$；防 λ 失控

---

## 3. Class Design

新增至 `src/pi_lnn/losses.py`：

```python
class AugmentedLagrangianMultiplier(nn.Module):
    """What: 單一 scalar penalty multiplier (λ, ρ) for one scalar constraint C ≥ 0.

    Why: continuity 是純 scalar、無 gauge 自由度。因為 C = mean((∇·u)²) ≥ 0，
         本實作其實是「accumulated-multiplier penalty schedule」（見 §2），不是
         textbook 對 signed g(x)=0 的 AL；λ 單調非負。命名仍沿用 AL 以對齊文獻
         慣例與外部 reviewer 用語。
         不繼承 nn.Parameter（λ 不靠 gradient 更新，靠 dual ascent）。
    """
    def __init__(
        self,
        init_lambda: float = 0.0,
        rho: float = 1.0,
        lambda_clip: float = 10.0,
        ema_momentum: float = 0.5,
    ) -> None:
        super().__init__()
        # 全部用 buffer（含 _initialized）→ state_dict 完整保存，resume 不會 EMA cold-start
        self.register_buffer("lambda_", torch.tensor(float(init_lambda)))
        self.register_buffer("ema_C", torch.tensor(0.0))
        self.register_buffer("_initialized", torch.tensor(False))
        # rho / lambda_clip / ema_momentum 為靜態 hyperparameter（不 schedule, v1）→ 用 Python float
        self.rho = float(rho)
        self.lambda_clip = float(lambda_clip)
        self.ema_momentum = float(ema_momentum)

    def loss_term(self, C: torch.Tensor) -> torch.Tensor:
        """λ·C + (ρ/2)·C² — primal-side differentiable term.

        Gradient 只流過 C（lambda_ 是 buffer 不接 autograd，rho 是 Python float）。
        """
        return self.lambda_ * C + 0.5 * self.rho * C ** 2

    @torch.no_grad()
    def update(self, C_batch: torch.Tensor) -> None:
        """Dual update：λ ← clip(λ + ρ·C̃, 0, Λ)。

        - C ≥ 0 always（squared divergence），故 lower clip = 0（負 λ 物理上無意義）
        - 用 out-of-place clamp 後寫回 lambda_，避免在 temp tensor 上 in-place 失效的 bug
        """
        c_val = C_batch.detach().reshape(()).to(self.ema_C.device, self.ema_C.dtype)
        if self.ema_momentum > 0.0 and bool(self._initialized.item()):
            self.ema_C.mul_(self.ema_momentum).add_(c_val * (1.0 - self.ema_momentum))
        else:
            self.ema_C.copy_(c_val)
            self._initialized.fill_(True)
        new_lambda = (self.lambda_ + self.rho * self.ema_C).clamp(0.0, self.lambda_clip)
        self.lambda_.copy_(new_lambda)
```

**Why 用 buffer 不用 Parameter**：λ 不接 autograd，更新由 dual ascent 控制，避免 optimizer state 混入 λ。

**修正紀錄（vs spec v1）**：
1. `clamp_()` in-place on temp tensor → 改為 out-of-place `.clamp(...)` 後 `.copy_()`（v1 BUG：clip 完全失效）。
2. `_initialized` 從 plain Python attribute → buffer（v1 BUG：resume 後 EMA 被清掉）。
3. clip range 從 `(-Λ, +Λ)` → `(0, Λ)`（v1 BUG：負 λ 對 C ≥ 0 無物理意義；對稱 clip 是 reviewer 抓出的概念錯誤）。
4. `c_val` 加 `.reshape(())` 強制 0-dim，防 caller 傳 [1] tensor 廣播污染。

---

## 4. Training Loop Integration

### Pre-conditions（runtime assert，不只 config-load）

於 `training.py` AL 啟用分支進入前一次性檢查（fail fast）：

```python
if use_al:
    assert args["lr_schedule"] != "ng", "AL incompatible with lr_schedule='ng'"
    assert continuity_weight == 0.0, \
        "AL active 時 continuity_weight 必須 = 0，否則 cont 被雙重 penalty"
    assert not args.get("use_sensor_physics", False), \
        "AL v1 不支援 use_sensor_physics（l_cont_total 會變成 sum-of-two-means，污染 EMA）"
    assert "cont" not in gradnorm_tasks, \
        "AL active 時 cont 必須從 gradnorm_tasks 移出（否則 GradNorm 與 AL 雙重控制）"
```

### First-order path（EXP-070/071 主路徑）

位置：`training.py` first-order path（line ~810+），在 `optimizer.zero_grad() → build l_total → backward → optimizer.step()` 序列改寫。

```python
# 既有：
# l_total = data_w * l_data + phys_weight * (l_ns + cont_w * l_cont) + ...
# l_total.backward(); optimizer.step()

# 改為（AL active 時）：
optimizer.zero_grad()
# ... 計算 l_data, l_ns_total, l_cont_total（與既有相同；l_cont_total 用 random colloc only）

if al_cont is not None:
    # primal: AL term 取代原本的 cont_w * l_cont
    al_term = al_cont.loss_term(l_cont_total)
    l_total = data_w * l_data + phys_weight * l_ns_total + al_term + ...
else:
    l_total = data_w * l_data + phys_weight * (l_ns_total + cont_w * l_cont_total) + ...

l_total.backward()
optimizer.step()

# dual update — 嚴格在 optimizer.step() 之後，且 step > 0 才動
if al_cont is not None and step > 0 and step % al_update_freq == 0:
    al_cont.update(l_cont_total.detach())
```

關鍵點：
- **時序鎖定**：`update()` 必在 `optimizer.step()` 之後執行 — 但**傳入的 `l_cont_total` 是 pre-step 計算的值**（forward 在 step 之前），這代表 dual update 使用的是「上一步參數估計的 C」。
  - 為何接受 pre-step C：post-step C 需要額外一次 forward + autograd（成本翻倍），對 every-100-step 的 dual update 不值得。
  - 一步延遲對 λ 軌跡影響可忽略（`al_update_freq=100` 的時間常數遠大於 1 step）。
  - 若未來想精確 post-step：可在 dual update 前重跑一次 inference-mode forward 算 C，但 v1 不做。
- **step=0 guard**：`step > 0 and step % freq == 0` — 隨機初始化的 C 無意義，不該污染 λ
- **`l_cont_total` 來源**：必須是 random-collocation 的 `mean(cont²)` 單一項，不是 `random + sensor_physics` 的和。spec §2 已強制 `use_sensor_physics=false`，此處的 `l_cont_total` 自動只剩 random colloc 的項
- **multi-RE 累加**：現有 code `l_cont_total = sum_i mean(cont_i²) / num_re`（dataset 層平均），此值即 AL 的 C，不需再除
- **`continuity_weight=0` 的安全網**：assert 已防錯設；即使誤設 `cont_w > 0`，由於 `l_total` 計算分支走 AL 路徑，`cont_w * l_cont` 不會出現

**EXP-071 path（GradNorm 同時 active 時）**：

```python
# v4 關鍵：AL term 不進 GradNorm losses 列表 — 完全 bypass GradNorm 的 mean_G aggregator
# gradnorm_tasks = ["data", "ns_u", "ns_v"]（3 元素，不含 "al"）

al_term = al_cont.loss_term(l_cont_total)           # λC + ρ/2·C²

# GradNorm 只看 3 個 non-AL tasks → mean_G 不含 G_al → w_ns_u/w_ns_v 不受 λ jump 擾動
gn_losses = [l_data, l_ns_u_total, l_ns_v_total]
_gradnorm_step(gn_weights, gn_losses, ref_params, ...)
gn_weights.normalize_to_data_()
w = gn_weights.weights                               # [w_d, w_u, w_v]

# AL term 以固定 weight = 1 加入；與 GradNorm 完全解耦
l_total = w[0] * l_data + w[1] * l_ns_u_total + w[2] * l_ns_v_total + al_term
l_total.backward(); optimizer.step()

if step > 0 and step % al_update_freq == 0:
    al_cont.update(l_cont_total.detach())            # dual update 用 raw C
```

**v4 設計變更**：v3 試圖以 `pin_("al", 1.0)` 把 AL term 從 GradNorm 「移除」，但 `_gradnorm_step` 內 `mean_G = mean(G_stack)` 會把 `G_al` 算進去，污染 `w_ns_u/w_ns_v` 的計算（reviewer B-V3-1）。v4 直接讓 `gn_losses` 不含 AL term，連 `pin_()` API 都不需要。EXP-071 與 EXP-070 的差別純粹在「GradNorm 是否平衡 data/ns 三 task」，AL 部分完全相同。

### LBFGS path

LBFGS closure 會被 line search 多次呼叫；λ 在 closure 內若變動會破壞 curvature 估計。

**契約**：
- closure 內：用當下 `al_cont.lambda_` 計算 `loss_term(C)`，**不**呼叫 `al_cont.update()`
- closure 外、`optimizer.step(closure)` 返回後：執行一次 `al_cont.update(l_cont_total_from_last_closure_call)`
- 由於 closure 多次呼叫，需在 closure 內 cache 最後一次的 `l_cont_total.detach()` 給外層 update 用

**LBFGS 與 GradNorm 互斥**：v4 不支援 `LBFGS + GradNorm + AL` 三者同時 active（會產生 closure 內 GradNorm weight 的 race condition）。pre-condition assert：

```python
if use_al and args["lr_schedule"] == "lbfgs":
    assert not use_gradnorm, "AL + LBFGS 不支援 use_gradnorm（closure race）"
```

EXP-070/071 都用 SOAP（`lr_schedule="soap"`），不觸發此限制。

### NG path

**不支援**：NG 內部用 residual 向量 + Gauss-Newton，AL 的 quadratic $\tfrac{\rho}{2}C^2$ 破壞此結構。
- Config-load 後（merge 完成）若 `lr_schedule == "ng" and use_augmented_lagrangian == true` → raise ValueError（在 `_validate_al_config` 中，§6）
- 訓練入口再次 assert（防 config 被 in-place mutate 後繞過）
- EXP-070/071 限用 SOAP（`lr_schedule="soap"`）或 AdamW（`lr_schedule="none"/"cosine"/"step"`）

---

## 5. GradNorm 互動

### 新增欄位與既有 5-task 自動偵測的關係

現況：`training.py:347` 用 `len(gradnorm_init_weights)==4 vs 5` 自動推斷是否含 BC task（`use_gradnorm_bc`）。新增 `gradnorm_tasks: list[str]` 後，**兩套機制必須協調**，避免雙重定義：

**規則**：
1. 若 config 提供 `gradnorm_tasks` 顯式指定 → 以此為準，**`gradnorm_init_weights` 長度必須相符**，否則 raise ValueError。
2. 若 config 未提供 `gradnorm_tasks`（向後相容）→ 由 `len(gradnorm_init_weights)` 推斷：
   - `len == 4` → `["data", "ns_u", "ns_v", "cont"]`
   - `len == 5` → `["data", "ns_u", "ns_v", "cont", "bc"]`（與既有 `use_gradnorm_bc` 自動偵測對齊）
   - 其他長度 → ValueError
3. 若 `gradnorm_tasks` 包含 `"bc"` → `use_gradnorm_bc=true` 自動派生（不需手動設）。
4. 既有 EXP 配置（`exp_064` 等）**完全無需改動** — shim 處理。

### 修改 `losses.py:GradNormWeights`

- `__init__` 加 `task_names: list[str]` 參數（向後相容預設 `None` → 由長度推斷）
- 新增 `index_of(name: str) -> int` API
- 新增 `__contains__(name: str) -> bool` API
- `_gradnorm_step` 邏輯不變（純數學），上層呼叫者依 `task_names` 動態組裝 `losses` 列表
- `normalize_to_data_()` 邏輯不變（仍以 index 0 = data 為 1 基準）

**注意**：v3 曾規劃 `pin_(name, value)` API 用於固定 `w_al`，但 v4 直接把 AL term 從 `gn_losses` 移除（避免 `mean_G` 污染），不再需要 pin。

### AL 與 GradNorm 同時 active 時 — AL term 如何進入 GradNorm balancing

**問題演進**：

- **v1 BLOCKER B2**：spec v1 把 AL term 加進 `l_total` 但 GradNorm 看不見它 → GradNorm 平衡基線錯誤。
- **v2 嘗試**：升 AL term 為 GradNorm 第 4 task → 引發 NM1（時間尺度錯配）+ NM2（dual update 失校）。
- **v3 嘗試**：保留 4-slot layout 但 `pin_(w_al)=1.0` → 仍有 cross-coupling leak（`_gradnorm_step` 內 `mean_G = mean(G_stack)` 把 `G_al` 算進去，污染 `w_ns_u/w_ns_v`）。
- **v4 最終**：**AL term 完全不進 `gn_losses` 列表**，GradNorm 只看 3 個 non-AL tasks。AL term 在 `gn_losses` 之外以 weight=1 加進 `l_total`。

**v4 規則**：

- 當 `use_gradnorm=true` AND `use_augmented_lagrangian=true`：
  - `gradnorm_tasks = ["data", "ns_u", "ns_v"]`（**3 元素**，不含 "al"）
  - `gradnorm_init_weights = [1.0, 0.01, 0.01]`（3 元素對應）
  - `_gradnorm_step` 的 `gn_losses = [l_data, l_ns_u, l_ns_v]`，`mean_G` 不含 G_al
  - AL term 在 `_gradnorm_step` 之外加進 `l_total`，effective coefficient = 1
  - 結果 loss = $w_d L_d + w_u L_{\text{ns},u} + w_v L_{\text{ns},v} + (\lambda C + \tfrac{\rho}{2}C^2)$
  - λ 的 dual update 單獨依 raw C，與 GradNorm 完全無耦合 — 真正解 NM1 / NM2 / B-V3-1
  - 代價：AL 與 data/ns 的相對比例由 λ 與 ρ 完全決定。這正是「AL 為主、GradNorm 處理 data/ns 平衡」的設計意圖
- 當 `use_gradnorm=false`（EXP-070 純 AL）：
  - 沒有 GradNorm，AL term 直接加進 `l_total`

**為何這次才對**：v2/v3 都試圖讓 GradNorm「以某種方式看見 AL」，但任何把 G_al 引入 `mean_G` 的設計都會把 λ 的 100-step jump 反向擾動 sibling weights。**唯一乾淨的解是讓 GradNorm 完全看不到 AL**。

### 飽和退化（已知模式）

當 λ 飽和到 $\Lambda$，AL term ≈ $\Lambda \cdot C + \tfrac{\rho}{2}C^2$；對小 C 而言主導項是 $\Lambda \cdot C$，等價於「fixed continuity weight = $\Lambda$」。**這是已知的退化模式**，不是要靠 EXP-071 才能發現的東西。EXP-071 真正測的是 **pre-saturation 區間**內 GradNorm 與 AL 是否相容；論文若觀察到飽和，需誠實標示「AL 在此 regime 退化為 static penalty」。

### Logging

現有 `training.py:474` 固定寬度 header `{'w_cont':>8}` 是位置敏感的。當 cont 被 AL 取代：
- log header 改為動態組裝：依 `gradnorm_tasks` + AL active 與否，列出 `w_<task>` 欄位 + 額外 `lambda_c` 欄位
- 既有非 AL 路徑保持 header 不變（向後相容）
- 此處變更同步寫入 `experiment_manifest.json` 的 `log_columns` field 供下游 evaluator 解析

具體 layout（AL active + GradNorm active 時，EXP-071）：

```
Step    L_data    L_phys  w_ns_u  w_ns_v  lambda_c   C_ema    L_total
   1   1.23e-2   4.56e-3   1.20    1.30   0.00e+0  0.00e+0   1.69e-2
 100   8.45e-3   2.31e-3   1.15    1.22   1.85e-1  1.85e-1   1.08e-2
```

EXP-070（AL on, GradNorm off）的 layout 同上但無 `w_ns_u/w_ns_v` 欄位。

欄位順序與寬度規則：
- `Step`: `>5`，`L_data`/`L_phys`/`L_total`: `>9` 科學記號
- 動態 `w_<task>`: `>7`（依 `gradnorm_tasks` 順序，跳過 `data`；v4 不含 `w_al` 因為 AL 不是 GradNorm task）
- `lambda_c`: `>9` 科學記號（AL active 時加）
- `C_ema`: `>9` 科學記號（AL active 時加，與 §8 primary indicator 對應）

非 AL 路徑保留既有 `{'w_cont':>8}` 等位置不變，向後相容。

---

## 6. Config Schema 新增

### TOML 欄位（必須同步加進 `src/pi_lnn/config.py:DEFAULT_LNN_ARGS`）

```toml
# AL 主控制
use_augmented_lagrangian = false  # 預設關閉，向後相容
al_init_lambda = 0.0
al_rho = 1.0
al_update_freq = 100              # 每 100 steps update 一次 λ
al_lambda_clip = 10.0
al_ema_momentum = 0.5

# GradNorm task layout（向後相容預設 [] → 由 init_weights 長度推斷）
gradnorm_tasks = []               # 例：["data","ns_u","ns_v"] 或 ["data","ns_u","ns_v","al"]
```

**移除 `al_constraints = [...]`**（spec v1 設計，reviewer 標 YAGNI）：v1 只有 continuity 一個 constraint，list 是過度設計，未來真要支援第二個 constraint 再 promote 為 list。

### `DEFAULT_LNN_ARGS` 必更新清單

`config.py:DEFAULT_LNN_ARGS`（line 9）採嚴格白名單（`config.py:204` raise on unknown keys）。新加的 7 個欄位若漏掉，**EXP-070..072 toml 連載入都會 ValueError**。

實作時必加：
```python
DEFAULT_LNN_ARGS = {
    # ... existing entries ...
    "use_augmented_lagrangian": False,
    "al_init_lambda": 0.0,
    "al_rho": 1.0,
    "al_update_freq": 100,
    "al_lambda_clip": 10.0,
    "al_ema_momentum": 0.5,
    "gradnorm_tasks": [],
}
```

### Semantic validation（new in `config.py:load_lnn_config`）

現有 `load_lnn_config` 只驗白名單，無 cross-field 語義檢查。為 AL 新增：

```python
def _validate_al_config(cfg: dict) -> None:
    """AL semantic validation — fail fast on the FULLY MERGED config.

    必須在 `DEFAULT_LNN_ARGS` 與 TOML 合併後呼叫（不能在 `load_lnn_config` 內），
    否則 `cfg["continuity_weight"]` 等欄位會拿到錯誤的 fallback。
    """
    if not cfg.get("use_augmented_lagrangian", False):
        # AL off: 仍要驗 gradnorm_tasks 不能誤刪 cont（否則 div constraint 消失）
        tasks = cfg.get("gradnorm_tasks", [])
        if cfg.get("use_gradnorm", False) and tasks and "cont" not in tasks \
                and cfg.get("continuity_weight", 0.0) > 0.0:
            raise ValueError(
                "use_gradnorm=True + cont not in gradnorm_tasks + continuity_weight>0 → "
                "cont 既不被 GradNorm 平衡也不在固定 weight loss 中（無效設定）"
            )
        return

    # AL on:
    # 注意：optimizer 種類由 lr_schedule 控制（config.py:128, training.py:213）
    if cfg.get("lr_schedule") == "ng":
        raise ValueError("use_augmented_lagrangian incompatible with lr_schedule='ng'")
    if cfg.get("lr_schedule") == "lbfgs" and cfg.get("use_gradnorm", False):
        raise ValueError("AL + LBFGS 不支援 use_gradnorm（closure race）")
    if cfg.get("continuity_weight", 0.0) != 0.0:
        raise ValueError(
            f"AL active 時 continuity_weight 必須 = 0，收到 {cfg['continuity_weight']}"
        )
    if cfg.get("use_sensor_physics", False):
        raise ValueError(
            "AL v1 不支援 use_sensor_physics（l_cont_total 會變成 sum-of-two-means）"
        )
    tasks = cfg.get("gradnorm_tasks", [])
    if tasks and "cont" in tasks:
        raise ValueError("AL active 時 'cont' 必須從 gradnorm_tasks 移出（即使 use_gradnorm=False）")
    if tasks and "al" in tasks:
        raise ValueError(
            "v4 規定 AL term 不進 GradNorm losses 列表 — 'al' 不能出現在 gradnorm_tasks"
        )
    init_w = cfg.get("gradnorm_init_weights", [])
    if tasks and len(init_w) != len(tasks):
        raise ValueError(
            f"gradnorm_init_weights 長度 ({len(init_w)}) 與 gradnorm_tasks ({len(tasks)}) 不符"
        )
```

**Call site**：`_validate_al_config` 必須在「`DEFAULT_LNN_ARGS` 與 TOML 合併之後」執行，不是在 `load_lnn_config` 內（後者只回傳 TOML keys，看不到 default fallback 值）。實作位置：`scripts/train_deeponet_cfc.py` 等入口處呼叫 `merged = {**DEFAULT_LNN_ARGS, **load_lnn_config(path)}; _validate_al_config(merged)`。

向後相容：`use_augmented_lagrangian = false`（預設）時所有 AL 邏輯短路，行為與目前完全一致 — `test_al_disabled_equivalence.py` 必驗。

---

## 7. EXP-070 / 071 / 072 Config 設計

### EXP-070 — Pure AL（無 GradNorm）

**Hypothesis:** AL（dynamic penalty schedule）本身能否獨立把 div L2 降下來？

```toml
# base = exp_064
lr_schedule = "soap"          # 或 "none"/"cosine"/"step"（AdamW），但不能 "ng"
use_gradnorm = false
data_loss_weight = 1.0
physics_loss_weight = 0.01    # 對應 EXP-064 GradNorm 收斂後的 ns_u/ns_v 量級（Open Q3）
continuity_weight = 0.0       # 必須 0（AL 接管），config-load 會 assert
poisson_loss_weight = 0.0
use_sensor_physics = false    # AL v1 強制（避免 l_cont_total = sum-of-two-means）

use_augmented_lagrangian = true
al_init_lambda = 0.0
al_rho = 1.0
al_update_freq = 100
al_lambda_clip = 10.0
al_ema_momentum = 0.5
```

### EXP-071 — AL + 3-task GradNorm（AL 完全在 GradNorm 之外）

**Hypothesis:** GradNorm 動態平衡 data/ns_u/ns_v 的同時，AL term 以 weight=1 加入 loss 是否與 EXP-070 結果一致？

```toml
# base = exp_064
lr_schedule = "soap"          # 與 EXP-070 對齊（不能 "ng"）
use_gradnorm = true
gradnorm_tasks = ["data", "ns_u", "ns_v"]         # 3 元素，不含 "al"
gradnorm_init_weights = [1.0, 0.01, 0.01]         # 3 元素對應
continuity_weight = 0.0
poisson_loss_weight = 0.0
use_sensor_physics = false

use_augmented_lagrangian = true
# AL 設定同 EXP-070
```

**v4 設計變更（vs v3）**：v3 用 `pin_(w_al)=1.0` 試圖解耦 AL 與 GradNorm，但 `_gradnorm_step` 內 `mean_G = mean(G_stack)` 仍會把 G_al 算進去 → 污染 `w_ns_u/w_ns_v` 的計算（架構 reviewer B-V3-1）。v4 直接讓 `gn_losses` 不含 AL term，GradNorm 完全看不到 G_al，從根本上消除耦合。EXP-070 vs EXP-071 的差別 = 「GradNorm 是否平衡 data/ns 三 task」，AL 部分完全相同。

### EXP-072 — Pressure Poisson 對照（無 AL）

**Hypothesis:** 壓 p 結構（Poisson）vs 壓 ∇·u（AL）哪個對 div 更有效？

```toml
# base = exp_064
use_gradnorm = true
gradnorm_tasks = ["data", "ns_u", "ns_v", "cont", "poisson"]
gradnorm_init_weights = [1.0, 0.01, 0.01, 0.01, 0.001]
continuity_weight = 1.0       # 保留原邏輯
poisson_loss_weight = 0.1     # 啟用 Poisson
use_sensor_physics = true     # EXP-064 既有設定

use_augmented_lagrangian = false
```

**Pressure Poisson 限制注意**：Poisson RHS 在 `physics.py:112` 是 $\nabla^2 p = -[(\partial_x u)^2 + (\partial_y v)^2 + 2(\partial_y u)(\partial_x v)]$，**推導時假設 $\nabla\cdot u = 0$ 已成立**。當前 baseline div_L2 ≈ 0.184，此 RHS 準確度為 $O(\nabla\cdot u)$；隨訓練 div 下降會自動修正。EXP-072 不會因此 ill-posed，但 early training 的 Poisson loss 是「近似方程」的 residual，這點要在 paper 寫清楚。

**`l_cont_total` 在 EXP-072 與 EXP-070/071 不同**：EXP-072 保留 `use_sensor_physics=true`，故 `l_cont_total` 是 `mean(cont²)_random + mean(cont_sp²)_sensor`（sum-of-two-means，EXP-064 既有結構）；EXP-070/071 則強制 `use_sensor_physics=false`，`l_cont_total` 只剩 random colloc 單一 mean。**Paper 比較三者的 div L2 時必須註明 cont 定義差異**，否則跨組 metric 不可直接比較。

---

## 8. Validation Criteria

| Metric | Baseline (EXP-064) | EXP-070/071 目標 | EXP-072 目標 |
|---|---|---|---|
| div L2 (primary) | 0.184 | **< 0.05** | < 0.10 |
| KE relative error | 7.80% | ≤ 9% (允許 trade-off) | ≤ 9% |
| u/v RMSE | 0.069/0.062 | ≤ 0.075/0.068 | ≤ 0.075/0.068 |
| **C_ema 收斂（AL primary indicator）** | — | `C_ema < 0.01 sustained 500 update steps` | — |
| λ_c 飽和監控（AL diagnostic） | — | log only — clip saturation 不視為收斂 | — |

### Primary indicator 改動原因（reviewer P2）

spec v1 用 `|λ_t - λ_{t-100}|/|λ_t| < 0.1` 作為收斂判準。但因為 $C \geq 0$ 永遠非負，λ 單調非減直到 hit `lambda_clip = 10.0`；該判準會**在 λ 飽和 clip 時觸發**，無論 constraint 是否真的滿足。**clip 飽和 ≠ 物理收斂**。

v2 改用 **C_ema** 作為主要收斂指標：
- C_ema < 0.01 持續 500 update steps → 視為 constraint 已壓住
- λ 的演化只作 diagnostic log，不直接判收斂
- 若 λ 在 C_ema 仍 > 0.05 時就已飽和 clip → log WARNING（信號：ρ 太小或 clip 太低）

### Stop early 條件

- div L2 連續 1000 steps 上升 ≥ 20% → log WARNING，不中止
- λ 飽和 clip 且 C_ema > 0.05 持續 500 update steps → log WARNING + 寫入 `experiment_manifest.json` `convergence_pathology` field
- 不自動中止訓練（讓最終結果說話）

---

## 9. Risks & Mitigations

| 風險 | 機率 | Mitigation |
|---|---|---|
| λ 失控爆炸 | 中 | `al_lambda_clip = 10.0` 硬限制 + `clamp(0, Λ)` 防負值 |
| **λ 飽和 clip 但 C_ema 仍大** | **高** | 監控 C_ema 為 primary indicator（§8）；若 λ=Λ 持續 500 update steps 而 C_ema > 0.05 → 降 ρ 10× 或設 `al_init_lambda = Λ/2` 給更多 headroom |
| ρ 太大 → 訓練震盪 | 中 | 預設 ρ=1.0，提供 ablation grid `[0.1, 1.0, 10.0]`；可後加 ρ schedule |
| ~~GradNorm 4-task 互相補償 / mean_G 污染~~ | — | **v4 已消除**：AL term 完全不進 `gn_losses`，GradNorm 看不到 `G_al`，根本上消除耦合（§5）|
| LBFGS closure 內 λ 變動破壞 line search | 中 | 強硬契約：closure 內僅讀 λ、不呼叫 `update()`；違反 → assert（test_al_multiplier 涵蓋）|
| Dual update 用 pre-step C（一步延遲） | 低 | `al_update_freq=100` 的時間常數遠大於 1 step；§4 已說明取捨 |
| Pressure Poisson 與 cont 雙重壓 p | 低 | EXP-072 與 EXP-070/071 完全分開，不混 |
| AL 在分散式/多 GPU 下 λ 不同步 | 低 | 單機訓練不影響；多機需 `all_reduce(C_batch)`，本 spec 不處理 |
| Resume from checkpoint 後 EMA cold-start | 低 | `_initialized` 已升級為 buffer（§3 修正），state_dict round-trip test 涵蓋 |
| `weights_only=True` 下 bool buffer 序列化 | 低 | 本 codebase 仍用 `weights_only=False`（既有 ckpt loader）；若未來升級，需 re-validate `_initialized` round-trip |
| EXP-070/071 vs EXP-072 cont 定義不同（前者 random-only，後者 sum-of-two-means） | 中 | §7 EXP-072 已註明；paper 比 div L2 時必須標註定義差異 |

---

## 10. Implementation Order

1. **`losses.py`**：
   - 加 `AugmentedLagrangianMultiplier` class（§3 修正版）
   - 改 `GradNormWeights` 加 `task_names: list[str]` 參數 + `index_of()` + `__contains__()`（向後相容預設 None → 由長度推斷）
2. **`config.py`** + **`scripts/train_deeponet_cfc.py`**（或對應 entry script）：
   - `config.py:DEFAULT_LNN_ARGS` 加 7 個新 key（§6 清單）
   - `config.py` 新增 `_validate_al_config(cfg)` 函式定義
   - **entry script** 在 `merged = {**DEFAULT_LNN_ARGS, **load_lnn_config(path)}` 之後呼叫 `_validate_al_config(merged)`（**不**在 `load_lnn_config` 內，後者只回 TOML keys 看不到 default fallback）
3. **`training.py`**：
   - `setup` 段：依 `use_augmented_lagrangian` 建立 `al_cont`；驗 pre-condition assert（§4）
   - first-order path：AL term 注入 + step > 0 guard + 嚴格 post-step dual update
   - LBFGS path：closure 內 cache `l_cont_total`，closure 外 update
   - Log：動態 header（§5 logging 段）
   - Manifest：寫入 `convergence_pathology` field
4. **Tests**（共 8 個，比 v1 多 5 個）：
   - `test_al_multiplier.py`：λ 更新數值（含 BUG-1 / BUG-3 regression）、clip(0, Λ) 行為、EMA 行為、step=0 guard
   - `test_al_checkpoint_roundtrip.py`：state_dict 含 `lambda_` / `ema_C` / `_initialized`；load 後 EMA 不 cold-start
   - `test_al_clip_boundary.py`：λ hit Λ 後不再增；C → 0 時 λ 凍結
   - `test_al_multi_re_normalization.py`：`l_cont_total` 在 multi-RE 下與單 RE 結果在尺度上 consistent（避免 num_re 倍誤差）
   - `test_al_ng_raise.py`：`use_al=true + lr_schedule="ng"` 在合併 config 後呼叫 `_validate_al_config` 時 raise ValueError；同時驗 `lbfgs + use_gradnorm` 也 raise
   - `test_gradnorm_task_names.py`：`gradnorm_tasks` 顯式指定 vs 由長度推斷的兩條路徑等價（4-task / 5-task / AL 3-task）；驗 `"al" in gradnorm_tasks` 會 raise
   - **`test_al_gradnorm_integration.py`**：EXP-071 路徑端到端 — `gn_losses` 只含 3 個 non-AL tasks（驗 `_gradnorm_step` 不 touch al_term）、`w_ns_u/w_ns_v/w_data` 動態更新、λ 獨立 dual update、loss 數值符合 `sum(w[i]*loss_i for i<3) + al_term`
   - `test_al_disabled_equivalence.py`：`use_augmented_lagrangian=false` 與既有行為 numerically equivalent（tol=1e-6，10 steps，**契約：AL 路徑不消耗 RNG**）
5. **Configs**：寫 EXP-070/071/072 三個 toml
6. **Smoke run**：每個 config 跑 `max(300, 3 * al_update_freq)` steps（預設 `al_update_freq=100` 時 = 300；若改 `freq=200` 則 = 600）。v1 寫 100，但 freq=100 時 0–1 次 update 不足以觀察 λ 演化。確認不 NaN、log 欄位齊全、AL `lambda_` 已動 ≥ 2 次
7. **Full run**：依算力安排 EXP-070 → 071 → 072 序列

---

## 11. Open Questions（送 owner）

1. **`al_update_freq = 100`** 是否合理？太短 λ 太雜訊；太長收斂慢。EXP-064 訓練長度約 10k–50k steps，100 對應 1%–0.2% epoch。
2. **ρ schedule**（隨 step 增加 ρ）？教科書 AL 常用，但會增加 hyperparameter。建議 v1 不做，作為 EXP-070 後 ablation。
3. **EXP-070 `physics_loss_weight = 0.01`** 從 EXP-064 GradNorm 收斂值估計，是否要先跑一個 short ablation 確認此值？最穩做法：先 run EXP-064 拿 final `w_ns_u/w_ns_v`，硬編碼進 EXP-070 config。
4. **EXP-071（AL + 3-task GradNorm）** 是否會出現 GradNorm 對 data/ns 的平衡與 AL term 的固定 weight=1 之間的尺度錯配？v4 後 GradNorm 看不到 AL，所以不會有 v2/v3 的 cross-coupling，但 GradNorm 的 `w_data/w_ns_u/w_ns_v` 仍可能被 AL 引發的 div/data trade-off 間接影響。這是 EXP-071 的核心測試項目；若 fail，fallback 是 EXP-070 結果單獨支撐主張。

---

## 12. 修訂歷程

- **v1 (2026-05-04)**：原始 spec
- **v2 (2026-05-04)**：依 3 個 subagent reviewer（physics-validation / architect-review / code-reviewer）回饋大幅修正，主要變更：
  - §3 BUG-1 修正：`clamp_()` no-op → out-of-place `.clamp()`（v1 clip 完全失效）
  - §3 BUG-2 修正：`_initialized` 升格 buffer（v1 resume 後 EMA cold-start）
  - §3 clip range：`(-Λ, +Λ)` → `(0, Λ)`（v1 對 C ≥ 0 概念錯誤）
  - §2 framing：「primal-dual」→「accumulated-multiplier penalty schedule」（誠實標示與 textbook AL 的差異）
  - §4 pre-condition runtime asserts + step=0 guard + 統一 update 在 optimizer.step() 之後 + l_cont_total 來源澄清（強制 use_sensor_physics=false）
  - §5 GradNorm 互動：AL term 升格為第 4 task（解 reviewer BLOCKER B2）+ 既有 4/5-task 自動偵測 backward-compat shim
  - §6 移除 YAGNI 的 `al_constraints` list；明列 `DEFAULT_LNN_ARGS` 必更新項；加 `_validate_al_config` semantic 檢查
  - §7 EXP configs 對齊 v2 規則 + Pressure Poisson approx 限制註記
  - §8 收斂 primary indicator：λ 穩定度 → C_ema sustained
  - §9 風險表加 LBFGS 契約 / clip 飽和 / resume cold-start 共 3 項
  - §10 tests 從 3 個擴成 7 個（補 checkpoint round-trip / NG raise / clip boundary / multi-RE）+ smoke 從 100 → 300 steps

- **v5 (2026-05-06)**：EXP-070..073 實驗證實 v1-v4 的 `use_sensor_physics=false` pre-condition 是錯誤假設。EXP-073 diagnostic（=EXP-064 完全相同 - 只關 sensor_physics）也產生與 EXP-070/070b/072 完全相同的場崩潰（u/v RMSE ~0.25, KE 84%, ω~0），證實 K=100 sensor 位置是 well-conditioned constraint 點，是場品質保住的核心。v5 反轉設計（Option 2）：
  - **§3 Pre-condition 翻轉**：`use_sensor_physics=true` 從「禁止」變「必要」
  - **§4 AL constraint 來源**：C = `mean(cont²)` at sensor positions only（well-conditioned subset），而非整個 random colloc 集合
  - **§4 dual update**：用 `l_cont_sensor_total`（與 `al_term` 同源），確保 primal-dual 一致性
  - **訓練 loop**：新增 `l_cont_sensor_total` accumulator 與 `l_cont_total` 並列；非 AL 路徑保持原行為（cont = random + sensor 兩者和）
  - **§6 `_validate_al_config`**：`use_sensor_physics=False` 時 raise（取代之前的 True 時 raise）
  - **§10 tests**：`test_al_validator.py` 改測 `without_sensor_physics_raises`；`_base_al_on()` fixture 改 `use_sensor_physics=True`
  - **EXP-074** 為 v5 首次驗證 run，clip=0.05 + sensor_physics=true + AL Option 2

- **v4 (2026-05-04)**：第三輪 architect-review 抓出 v3 的 BLOCKER B-V3-1 — `pin_(w_al)=1.0` 看似解耦但 `_gradnorm_step` 內 `mean_G = mean(G_stack)` 仍含 `G_al`，污染 `w_ns_u/w_ns_v` 計算。v4 修補：
  - **§5 / §4 / §7 EXP-071**：AL term **完全不進 `gn_losses` 列表**。`gradnorm_tasks = ["data","ns_u","ns_v"]`（3 元素），AL term 在 GradNorm 之外以 weight=1 加進 `l_total`。從根本上消除 `G_al → mean_G → sibling weights` 的耦合鏈
  - **§5**：移除 `GradNormWeights.pin_(name, value)` API（v3 設計，不再需要）
  - **§5 logging**：layout 移除 `w_al` 欄位（v4 後 AL 不是 GradNorm task，不顯示為 weight）
  - **§4 LBFGS**：補 `LBFGS + GradNorm + AL` 三者互斥的 pre-condition assert（解 reviewer M-V3-2）
  - **§6 `_validate_al_config`**：
    - 加 `lr_schedule="lbfgs" + use_gradnorm=true + AL` 三者衝突檢查
    - 加 `"al" in gradnorm_tasks` 禁止規則（v4 不允許）
    - 加「AL off + 3-task gradnorm + cont 缺席 + continuity_weight>0」無效設定守門（解 reviewer Mn-V3-2）
  - **§10 step 2**：明確標示 `_validate_al_config` 在 entry script merge 後呼叫，不在 `load_lnn_config` 內（修 §10 vs §6 矛盾，解 reviewer Mn-V3-3）
  - **§10 tests**：rename `test_gradnorm_al_4task_integration.py` → `test_al_gradnorm_integration.py`；`test_al_ng_raise.py` 加 LBFGS+GradNorm 案例；`test_gradnorm_task_names.py` 加 `"al" in tasks` raise 驗證；移除 `pin_()` 相關測試

- **v3 (2026-05-04)**：再經 3 個 subagent reviewer 對 v2 的二次審查，修補 v2 引入的新問題：
  - **§6 BLOCKER 修正**：`_validate_al_config` 中 `cfg.get("optimizer")` → `cfg.get("lr_schedule")`（v2 用了不存在的 key 名，validation 形同虛設）
  - **§6 call site 澄清**：明確標示 validator 必須在 `DEFAULT_LNN_ARGS` 與 TOML 合併**之後**呼叫，不能在 `load_lnn_config` 內（後者只回 TOML keys）
  - **§7 EXP-070 修正**：`optimizer = "adamw"` → `lr_schedule = "soap"`（同上 key 名問題）
  - **§5 / §7 EXP-071 重大設計變更**：v2 把 AL term 升為 GradNorm 第 4 task 引發 NM1（時間尺度錯配）+ NM2（dual update 失校）→ v3 採物理 reviewer 建議 (b)：**`w_al` 永久 pin = 1.0**，GradNorm 只動 `data/ns_u/ns_v` 三權重，AL 與 GradNorm 完全解耦
  - §5 新增 `GradNormWeights.pin_(name, value)` API
  - §4 修正 prose vs pseudo-code 不一致：明確標示 dual update 用 **pre-step C**（一步延遲，可接受）；新增 EXP-071 path 完整 pseudo-code 顯示 `w_al` pin 整合
  - §5 飽和退化「reframe」：從「EXP-071 的目的就是測這個」→「已知退化模式；EXP-071 測 pre-saturation 區間」
  - §7 EXP-072 新增 cont 定義差異註記（vs EXP-070/071）
  - §5 logging：補具體 header layout（Step / L_data / L_phys / w_<task> / lambda_c / C_ema / L_total）
  - §9 風險表：移除 v2 的「w_al/λ 互相補償」（已被 v3 pin 解決）；新增 pre-step C 一步延遲、weights_only=True 序列化、cont 定義跨組差異
  - §10 tests：v2 的 7 個 → v3 的 8 個（新增 `test_gradnorm_al_4task_integration.py`）；`test_al_ng_raise.py` 改用 `lr_schedule` 而非 `optimizer`；smoke 算式表達修正
  - §10 step 6 smoke 計算式：移除 v2 寫死的 `= 300`（誤導：當 freq≠100 時不成立）

決策後即可進入實作。
