"""一次性生成「對等化 ablation」config（EXP-281/282/283 × seed{42,1,2,3,4} @ 20k）。

What: 把 B0/B1/B2 (EXP-246/247/248, 10k single-seed) 升級成 20k × n=5，與 B3 (EXP-245) 對齊。
Why : Table 4.6 的 ablation 目前 B3=20k×n=5 vs B0/B1/B2=10k×single，training budget 不對等，
      使 contribution#1「cross-attention 是 dominant lever」無法分離「架構增益 vs 訓練預算」。
改動: iterations 10000→20000；加 time_marching_warmup_steps=2000（對齊 B3 fixed-step warmup，
      否則 0.3×20000=6000 步又成新變因）；seed；artifacts_dir；header 註解。其餘欄位逐字保留。
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = ROOT / "configs"
STABLE = CFG / "stable"

# (exp_id, base_filename, variant_label, variant_low, artifacts_prefix, legacy_ref)
SPECS = [
    ("281", "exp_246_b0_les_T50.toml", "B0 (vanilla DeepONet)",      "b0", "deeponet-re10000",     "EXP-246"),
    ("282", "exp_247_b1_les_T50.toml", "B1 (CfC, no cross-attn)",    "b1", "deeponet-cfc-re10000", "EXP-247"),
    ("283", "exp_248_b2_les_T50.toml", "B2 (cross-attn, no CfC)",    "b2", "deeponet-cfc-re10000", "EXP-248"),
]
# (filename_suffix_char, seed, stable_letter)
SEEDS = [("", 42, "a"), ("b", 1, "b"), ("c", 2, "c"), ("d", 3, "d"), ("e", 4, "e")]

made = []
for eid, base_fn, vlabel, vlow, prefix, legacy in SPECS:
    base_text = (CFG / base_fn).read_text()
    for sufchar, seed, slet in SEEDS:
        if sufchar == "":
            fname = f"exp_{eid}_{vlow}_les_T50_20k.toml"
            adir = f"artifacts/kolmogorov/{prefix}-exp{eid}-{vlow}-les-T50-20k"
        else:
            fname = f"exp_{eid}{sufchar}_{vlow}_les_T50_20k_seed{seed}.toml"
            adir = f"artifacts/kolmogorov/{prefix}-exp{eid}{sufchar}-{vlow}-les-T50-20k-seed{seed}"

        text = base_text
        # 1) iterations
        assert "iterations = 10000" in text, f"{base_fn}: 找不到 iterations=10000"
        text = text.replace("iterations = 10000", "iterations = 20000")
        # 2) fixed-step warmup（對齊 B3）
        assert "time_marching_warmup = 0.3\n" in text, f"{base_fn}: 找不到 time_marching_warmup"
        text = text.replace(
            "time_marching_warmup = 0.3\n",
            "time_marching_warmup = 0.3\n"
            "time_marching_warmup_steps = 2000   # fixed-step, 對齊 EXP-245 (B3) training budget\n",
        )
        # 3) seed
        if seed != 42:
            assert "seed = 42" in text, f"{base_fn}: 找不到 seed=42"
            text = text.replace("seed = 42", f"seed = {seed}")
        # 4) artifacts_dir（整行替換）
        text = re.sub(r'artifacts_dir = ".*"', f'artifacts_dir = "{adir}"', text)
        # 5) header：替換 [train] 之前的所有註解
        idx = text.index("[train]")
        header = (
            f"# configs/{fname}\n"
            f"# What: EXP-{eid}_{slet} — {vlabel} + LES_T50, 20k iter, seed {seed}\n"
            f"# Why : 對等化 ablation。B3(EXP-245)=20k×n=5，原 {legacy}=10k×single seed，\n"
            f"#       training budget 不對等使 contribution#1「cross-attention dominant」無法分離架構 vs 預算。\n"
            f"#       本系列補 B0/B1/B2 各 20k×n=5(seed 42/1/2/3/4)，與 EXP-245 完全對齊。\n"
            f"# 派生 from {legacy}；改動：iterations→20000、加 time_marching_warmup_steps=2000、seed、artifacts_dir。\n\n"
        )
        text = header + text[idx:]

        out = CFG / fname
        out.write_text(text)
        made.append((eid, slet, seed, fname, adir))

        # stable symlink: exp_<eid>_<slet>.toml -> ../<fname>
        link = STABLE / f"exp_{eid}_{slet}.toml"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(pathlib.Path("..") / fname)

print(f"已生成 {len(made)} 個 config + stable symlink：")
for eid, slet, seed, fname, adir in made:
    print(f"  EXP-{eid}_{slet} (seed={seed:>2})  {fname}")
