#!/bin/bash
# scripts/slurm/eval_kscaling_spectra.sbatch
# What: 重出 K-scaling 能譜圖 (fig:k_scaling_spectra 的 spectrum_K100/200/400)，
#       目的只有一個 —— 讓參考線從 k^{-5/3} 換成正確的 k^{-3}
#       (2D forward enstrophy-cascade, k>k_f；k^{-5/3} 是 k<k_f 的 inverse cascade)。
# Why: evaluate_deeponet_cfc.py 的參考線已於本機改成 k^{-3}（plot_energy_spectrum）。
#      spectrum_K* 需要重跑 eval 才會更新；K=200/400 checkpoint 只在 lab-server。
#      eval 輸出 <output-dir>/energy_spectrum.pdf，內含 DNS + PI-CON 譜 + k^{-3} 參考
#      + 自動的 sensor Nyquist 線 sqrt(K/pi)。跑完 copy/rename 成 spectrum_K<K>_nyquist.pdf。
#SBATCH --job-name=eval_kscaling_spectra
#SBATCH --output=logs/eval_kscaling_spectra_%j.out
#SBATCH --error=logs/eval_kscaling_spectra_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pi-lnn}"

export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；先在 head node 跑 'uv sync --python 3.12'" >&2; exit 3; }

OUT="artifacts/kolmogorov/kscaling_spectra"
mkdir -p "$OUT"

# ─────────────────────────────────────────────────────────────────────────────
# 三個 K-scaling run：K | config | checkpoint
#   - K=100 (EXP-245) 路徑已在本機驗證，可直接跑。
#   - K=200 (EXP-269) / K=400 (EXP-270) checkpoint 只在 lab-server，路徑請「務必核對」。
#     若你的 EXP-269/270 有專屬 config，把 CONFIG 換掉；沒有就沿用 256/257 config
#     （eval 只從 config 讀 K / k_f / DNS / sensor json，iterations 不影響 eval）。
#   格式： "K|CONFIG_PATH|CHECKPOINT_PATH"
# ─────────────────────────────────────────────────────────────────────────────
RUNS=(
  "100|configs/exp_245_b3_les_T50.toml|artifacts/kolmogorov/deeponet-cfc-re10000-exp245-b3-les-T50-20k/picon_kolmogorov_final.pt"
  "200|configs/exp_256_les_K200.toml|__VERIFY_EXP269_K200_20k__/picon_kolmogorov_final.pt"   # ⚠ 換成 EXP-269 (K=200, 20k) 的 artifacts_dir
  "400|configs/exp_257_les_K400.toml|__VERIFY_EXP270_K400_20k__/picon_kolmogorov_final.pt"   # ⚠ 換成 EXP-270 (K=400, 20k) 的 artifacts_dir
)

echo "======================================================================"
echo "[eval_kscaling_spectra] $(date '+%F %T')  slurm_job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}"
nvidia-smi -L 2>/dev/null || echo "[WARN] nvidia-smi 不可用"
echo "======================================================================"

fail=0
for row in "${RUNS[@]}"; do
  K="${row%%|*}"; rest="${row#*|}"
  CFG="${rest%%|*}"; CKPT="${rest#*|}"
  edir="${OUT}/K${K}-eval"
  echo ""
  echo "############### [eval] K=${K}  $(date '+%F %T') ###############"
  echo "  config: ${CFG}"
  echo "  ckpt:   ${CKPT}"
  if [ ! -f "$CKPT" ]; then
    echo "[ERR] checkpoint 不存在: ${CKPT}（K=${K} 跳過；請核對路徑）" >&2
    fail=1; continue
  fi
  uv run python -u scripts/evaluate_deeponet_cfc.py \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --output-dir "$edir" \
    --device cuda
  # 收集：eval 產出的 energy_spectrum.pdf → 論文檔名 spectrum_K<K>_nyquist.pdf
  if [ -f "${edir}/energy_spectrum.pdf" ]; then
    cp "${edir}/energy_spectrum.pdf" "${OUT}/spectrum_K${K}_nyquist.pdf"
    cp "${edir}/energy_spectrum.png" "${OUT}/spectrum_K${K}_nyquist.png" 2>/dev/null || true
    echo "  → ${OUT}/spectrum_K${K}_nyquist.pdf"
  else
    echo "[ERR] eval 未產出 energy_spectrum.pdf（K=${K}）" >&2; fail=1
  fi
done

echo ""
echo "======================================================================"
echo "[$(date '+%F %T')] 完成。收集於 ${OUT}/spectrum_K{100,200,400}_nyquist.pdf"
echo ""
echo "回本機更新論文圖（在你的 Mac 上跑，thesis/ 是 gitignore、不在 lab-server）："
echo "  scp lab-server:~/pi-lnn/${OUT}/spectrum_K100_nyquist.pdf thesis/figures/results/"
echo "  scp lab-server:~/pi-lnn/${OUT}/spectrum_K200_nyquist.pdf thesis/figures/results/"
echo "  scp lab-server:~/pi-lnn/${OUT}/spectrum_K400_nyquist.pdf thesis/figures/results/"
echo "======================================================================"
[ "$fail" = "0" ] || { echo "[WARN] 有 K 未成功（見上）；修正路徑後重送。" >&2; exit 1; }
