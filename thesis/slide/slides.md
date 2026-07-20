---
theme: default
title: Physics-Constrained Continuous-Time Reconstruction of Turbulent Flows
  from Sparse Sensors
info: |
  Thesis Defense, Junyi Li, final version
class: text-left
colorSchema: light
fonts:
  sans: "Arial"
  mono: "JetBrains Mono"
transition: fade
mdc: true
download: false
exportFilename: pi-lnn-talk
---

<!-- ====================================================================
     SLIDE 01 · COVER  (pptx Cover layout: 無 chevron NavBar，只有 logos)
     ==================================================================== -->

<div class="absolute top-12 left-16 right-16">
  <SectionTag>Master Thesis Defense · Engineering and System Science, NTHU · July 2026</SectionTag>
</div>

<div class="absolute left-16 right-16" style="top: 30%;">

<div class="text-base mb-2" style="color:#6B7280;">物理約束之稀疏感測器連續時間湍流場重建</div>

<h1 style="font-size: 2.2rem; line-height: 1.15; font-weight: 700; color: #7F1084; letter-spacing: -0.01em;">
Physics-Constrained Continuous-Time<br/>Reconstruction of Turbulent Flows from <br>Sparse Sensors
</h1>

<div class="mt-5 text-sm" style="color:#4B5563;">
  2-D Kolmogorov flow at <b style="color:#7F1084;">Re = 10⁴</b>,
  reconstructed from <b style="color:#7F1084;">100 velocity sensors</b>,
  with Navier–Stokes (NS) residual as the only physics signal —
  no Direct Numerical Simulation (DNS) supervision in training.
</div>

<div class="mt-8" style="display:grid; grid-template-columns:max-content 1fr; column-gap:14px; row-gap:4px; align-items:baseline; font-size:0.9rem;">
  <span style="color:#9CA3AF;">Presenter</span><span style="color:#1F1B2E;"><b>李駿毅</b> Jun-Yi Li <span style="color:#9CA3AF;">, 113011527</span></span>
  <span style="color:#9CA3AF;">Advisor</span><span style="color:#1F1B2E;"><b>林洸銓</b> Dr. Kuang C. Lin</span>
  <span style="color:#9CA3AF;">Lab</span><span style="color:#6B7280;">Applied Computing &amp; Thermofluid Laboratory</span>
</div>

</div>

<FooterLogos />

<!--
[封面 · 30s]
• Anchor：K=100 sensors／NS residual 是唯一 physics／訓練不看 DNS
• 大綱：問題 → 架構 → 訓練 → 結果 → 限制
-->

---

<NavBar active="background" />

<SectionTag>§ Background · why reconstruction is needed</SectionTag>

# Why the field has to be reconstructed

<style>
.recon-bg { display:grid; grid-template-columns:34% minmax(0,1fr); gap:28px; margin-top:16px; align-items:center; }
.recon-bg .lead { font-size:1.18rem; line-height:1.38; color:#1F1B2E; }
.recon-bg .lead b { color:#7F1084; }
.recon-bg .reason { margin-top:20px; display:grid; gap:10px; }
.recon-bg .reason > div { display:grid; grid-template-columns:30px minmax(0,1fr); align-items:start; font-size:.86rem; line-height:1.35; color:#374151; }
.recon-bg .reason > div > div { display:block; }
.recon-bg .reason span { color:#E97132; font-weight:700; }
.recon-flow { display:grid; grid-template-columns:1fr 44px 1.08fr 44px 1fr; align-items:center; min-width:0; }
.recon-flow .stage { min-height:198px; border:1px solid #E5E0EC; border-radius:12px; background:rgba(255,255,255,.78); padding:16px 14px; }
.recon-flow .tag { font-size:.70rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:#6B7280; }
.recon-flow .big { font-size:1.05rem; font-weight:700; line-height:1.25; margin-top:8px; color:#1F1B2E; }
.recon-flow .small { font-size:.79rem; line-height:1.35; color:#6B7280; margin-top:8px; }
.recon-flow .arrow { text-align:center; color:#7F1084; font-size:1.8rem; font-weight:700; }
.recon-flow .samples { height:78px; position:relative; margin-top:12px; border-bottom:1px solid #D8D2E0; }
.recon-flow .samples i { position:absolute; width:8px; height:8px; border-radius:50%; background:#E97132; }
.recon-flow .field { height:78px; margin-top:12px; border-radius:7px; background:radial-gradient(circle at 28% 42%,#E8C6EA 0 10%,transparent 11%),radial-gradient(circle at 67% 60%,#C6DDEA 0 13%,transparent 14%),linear-gradient(135deg,#F8EFF8,#E8F4F7); border:1px solid #E5E0EC; }
.recon-flow .tool { background:#FAF2FB; border-color:#7F1084; display:flex; flex-direction:column; justify-content:center; text-align:center; }
</style>

<div class="recon-bg">
  <div>
    <div class="lead">Most instruments observe a system only at <b>selected locations or times</b>, while interpretation requires the state of the <b>whole domain</b>.</div>
    <div class="reason">
      <div><span>01</span><div>Measurements are local and incomplete.</div></div>
      <div><span>02</span><div>Quantities of interest depend on spatial structure between measurements.</div></div>
      <div><span>03</span><div>Reconstruction supplies a coherent field estimate for analysis and decision-making.</div></div>
    </div>
  </div>

  <div class="recon-flow">
    <div class="stage">
      <div class="tag">Available</div><div class="big">Partial measurements</div>
      <div class="samples"><i style="left:12%;top:48%;"></i><i style="left:36%;top:20%;"></i><i style="left:61%;top:58%;"></i><i style="left:84%;top:31%;"></i></div>
      <div class="small">isolated samples of an evolving system</div>
    </div>
    <div class="arrow">→</div>
    <div class="stage tool"><div class="tag" style="color:#7F1084;">Tool</div><div class="big" style="color:#7F1084;font-size:1.25rem;">Reconstruction</div><div class="small">infer what happens between observations</div></div>
    <div class="arrow">→</div>
    <div class="stage">
      <div class="tag">Required</div><div class="big">Continuous field</div><div class="field"></div>
      <div class="small">structures, gradients, and system-wide quantities</div>
    </div>
  </div>
</div>

<FooterLogos />

<!--
[為何需要重建 · 1min]
• 儀器只量得到局部，工程決策要整個場
⚠️ 只講背景，不提 K / DNS / NS
-->

---

<NavBar active="background" />

<SectionTag>§ Background · what a PINN does</SectionTag>

# Physics-informed neural networks

<div class="text-xs mt-1" style="color:#374151;">
The network <b>is</b> the field: training adjusts θ from data and physics; inference reuses the frozen θ in one forward pass.
</div>

<div class="mt-1">
</div>

<svg class="pinn-fixed" viewBox="0 0 880 312" style="width:100%;height:auto;margin-top:8px;">
  <defs>
    <marker id="pinn-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#7F1084"/>
    </marker>
  </defs>

  <rect x="1" y="1" width="878" height="203" rx="11" fill="rgba(255,255,255,.78)" stroke="#DDD6E5"/>
  <text x="16" y="25" fill="#7F1084" style="font-size:12px;font-weight:700;letter-spacing:.07em;">01 TRAINING</text>
  <text x="108" y="25" fill="#6B7280" style="font-size:11px;">evaluate data and PDE residuals, then update the weights</text>

  <g style="font-family:Inter,Arial,sans-serif;">
    <rect x="18" y="72" width="116" height="54" rx="7" fill="#FFF" stroke="#D8D2E0"/>
    <text x="76" y="95" text-anchor="middle" fill="#1F1B2E" style="font-size:12px;font-weight:700;">Coordinates</text>
    <text x="76" y="113" text-anchor="middle" fill="#6B7280" style="font-size:10px;">(x, y, t)</text>
    <path d="M134 99 H177" fill="none" stroke="#7F1084" stroke-width="1.5" marker-end="url(#pinn-arrow)"/>
    <rect x="184" y="72" width="154" height="54" rx="7" fill="#F4F6F9" stroke="#0F2D52"/>
    <text x="261" y="95" text-anchor="middle" fill="#1F1B2E" style="font-size:12px;font-weight:700;">Neural field N(·; θ)</text>
    <text x="261" y="113" text-anchor="middle" fill="#6B7280" style="font-size:10px;">trainable weights θ</text>
    <path d="M199 62 V54 H323 V62" fill="none" stroke="#0F2D52" stroke-width="1"/>
    <text x="261" y="48" text-anchor="middle" fill="#0F2D52" style="font-size:10px;font-weight:700;">n layers × m neurons</text>
    <path d="M338 99 H376" fill="none" stroke="#7F1084" stroke-width="1.5" marker-end="url(#pinn-arrow)"/>
    <rect x="383" y="72" width="116" height="54" rx="7" fill="#FFF" stroke="#D8D2E0"/>
    <text x="441" y="95" text-anchor="middle" fill="#1F1B2E" style="font-size:12px;font-weight:700;">Prediction</text>
    <text x="441" y="113" text-anchor="middle" fill="#6B7280" style="font-size:10px;">u, v, p</text>
    <path d="M499 99 H526 V65 H548" fill="none" stroke="#7F1084" stroke-width="1.5" marker-end="url(#pinn-arrow)"/>
    <path d="M526 99 V140 H548" fill="none" stroke="#7F1084" stroke-width="1.5" marker-end="url(#pinn-arrow)"/>
    <circle cx="526" cy="99" r="3" fill="#7F1084"/>
    <rect x="555" y="40" width="142" height="51" rx="6" fill="#FEF6F1" stroke="#E9C9B2"/>
    <text x="566" y="59" fill="#E97132" style="font-size:10px;font-weight:700;letter-spacing:.04em;">DATA RESIDUAL</text>
    <text x="566" y="77" fill="#374151" style="font-size:10px;">prediction vs. measurements</text>
    <rect x="555" y="115" width="142" height="51" rx="6" fill="#FEF6F1" stroke="#E9C9B2"/>
    <text x="566" y="134" fill="#E97132" style="font-size:10px;font-weight:700;letter-spacing:.04em;">PDE RESIDUAL</text>
    <text x="566" y="152" fill="#374151" style="font-size:10px;">autodiff → equations</text>
    <path d="M697 65 H724 V83 H742" fill="none" stroke="#7F1084" stroke-width="1.5" marker-end="url(#pinn-arrow)"/>
    <path d="M697 140 H724 V103 H742" fill="none" stroke="#7F1084" stroke-width="1.5" marker-end="url(#pinn-arrow)"/>
    <rect x="749" y="66" width="111" height="55" rx="7" fill="#FAF2FB" stroke="#7F1084"/>
    <text x="804.5" y="89" text-anchor="middle" fill="#7F1084" style="font-size:12px;font-weight:700;">Total loss</text>
    <text x="804.5" y="107" text-anchor="middle" fill="#6B7280" style="font-size:10px;">combine residuals</text>
    <path d="M804 121 V139" fill="none" stroke="#7F1084" stroke-width="1.5" marker-end="url(#pinn-arrow)"/>
    <rect x="749" y="146" width="111" height="38" rx="7" fill="#FFF" stroke="#7F1084"/>
    <text x="804.5" y="169" text-anchor="middle" fill="#7F1084" style="font-size:12px;font-weight:700;">Optimizer</text>
    <path d="M749 165 H710 V190 H261 V133" fill="none" stroke="#7F1084" stroke-width="1.5" marker-end="url(#pinn-arrow)"/>
    <text x="484" y="186" text-anchor="middle" fill="#7F1084" style="font-size:10px;font-weight:700;">update θ</text>
  </g>

  <rect x="1" y="219" width="878" height="87" rx="11" fill="rgba(255,255,255,.78)" stroke="#DDD6E5"/>
  <text x="16" y="242" fill="#0F2D52" style="font-size:12px;font-weight:700;letter-spacing:.07em;">02 INFERENCE</text>
  <text x="119" y="242" fill="#6B7280" style="font-size:11px;">freeze θ*; no loss, optimizer, or back-propagation</text>
  <rect x="22" y="254" width="210" height="39" rx="7" fill="#FFF" stroke="#D8D2E0"/>
  <text x="127" y="278" text-anchor="middle" fill="#1F1B2E" style="font-size:12px;font-weight:700;">Query coordinate (x, y, t)</text>
  <path d="M232 273 H303" fill="none" stroke="#7F1084" stroke-width="1.5" marker-end="url(#pinn-arrow)"/>
  <rect x="310" y="254" width="260" height="39" rx="7" fill="#FAF2FB" stroke="#7F1084"/>
  <text x="440" y="278" text-anchor="middle" fill="#7F1084" style="font-size:12px;font-weight:700;">Trained field N(·; θ*)</text>
  <path d="M570 273 H641" fill="none" stroke="#7F1084" stroke-width="1.5" marker-end="url(#pinn-arrow)"/>
  <rect x="648" y="254" width="210" height="39" rx="7" fill="#FFF" stroke="#D8D2E0"/>
  <text x="753" y="278" text-anchor="middle" fill="#1F1B2E" style="font-size:12px;font-weight:700;">Field value (u, v, p)</text>
</svg>

<FooterLogos />

<!--
[PINN 怎麼運作 · 1min]
• Training：座標 → 預測 → data + PDE residual → 更新 θ
• Inference：θ 凍結，只送座標取值
⚠️ 這是 PINN 通論，不是 PI-CON
⚠️ 圖中層數寬度為示意
-->

---

<NavBar active="literature" />

<SectionTag>§ Literature review · classical sparse-sensor reconstruction</SectionTag>

# What sparse-sensor reconstructions assume

<div class="text-sm mt-1" style="color:#374151;">
At each new time, every method receives the same current K = 100 sensor values <span class="raw">y(t)</span>. The table isolates the prior that differentiates them.
</div>

<style>
.cl { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 8px; }
.cl th { text-align: left; font-weight: 700; color: #9CA3AF; font-size: 0.68rem; text-transform: uppercase;
         letter-spacing: 0.05em; padding: 0 10px 6px 10px; border-bottom: 1px solid #D8D2E0; }
.cl th.x { text-align: center; }
.cl td { padding: 6px 10px; border-bottom: 1px solid #F1EDF5; color: #374151; vertical-align: middle; }
.cl td.who { color: #1F1B2E; font-weight: 600; white-space: nowrap; }
.cl td.src { color: #9CA3AF; font-size: 0.82em; }
.cl td.x { text-align: center; font-weight: 700; font-size: 1.05rem; }
.cl .no { color: #E97132; }
.cl tr.grp td { border-bottom: none; padding-top: 7px; padding-bottom: 1px;
                font-size: 0.71rem; font-weight: 700; letter-spacing: 0.05em;
                text-transform: uppercase; color: #1F1B2E; }
</style>

<table class="cl">
<thead>
<tr>
<th style="width: 34%;">Method</th>
<th style="width: 40%;">Field-filling prior</th>
<th style="width: 13%;" class="x">Full-field<br/>library?</th>
<th style="width: 13%;" class="x">Physics<br/>constraint?</th>
</tr>
</thead>
<tbody>

<tr class="grp"><td colspan="4">Sensor-only interpolation</td></tr>
<tr><td class="who">RBF <span style="font-size:.78em; color:#9CA3AF; font-weight:400;">[Hardy 1971]</span></td><td>local smoothness; radial kernel (<span class="raw">ε</span> = 10)</td><td class="x">no</td><td class="x">no</td></tr>
<tr><td class="who">IDW <span style="font-size:.78em; color:#9CA3AF; font-weight:400;">[Shepard 1968]</span></td><td>nearer sensors receive higher weight (1/d²)</td><td class="x">no</td><td class="x">no</td></tr>

<tr class="grp"><td colspan="4">Constrained basis reconstruction</td></tr>
<tr><td class="who">Divergence-free<br/>trig-LSQ <span style="font-size:.70em; color:#9CA3AF; font-weight:400;">[Fourier basis: Boyd 2001]</span></td><td>diagnostic baseline: divergence-free modes, <span class="raw">k ≤ 5</span></td><td class="x">no</td><td class="x no">yes</td></tr>

<tr class="grp"><td colspan="4">Reduced-order model</td></tr>
<tr><td class="who">Gappy-POD <span style="font-size:.78em; color:#9CA3AF; font-weight:400;">[Everson &amp; Sirovich 1995]</span></td><td class="no">r POD modes (r = 100 here); learned from 160 DNS fields</td><td class="x no">yes</td><td class="x">no</td></tr>

</tbody>
</table>

<FooterLogos />

<!--
[稀疏量測重建的先驗 · 1.5min]
• RBF/IDW 是 sensor-only interpolation；trig-LSQ 是帶無散低頻基底的 diagnostic constrained reconstruction
• 三者都只用當下 y(t)，差別在平滑、距離、或 low-band divergence-free 先驗
• Gappy-POD：offline full fields 先形成 Φ_r；online 用當下 y(t) fit coefficients，再重建該時刻
• 因此 Gappy 的 r=100 是 DNS-basis oracle；不是 t=0→5 forecast，也不是公平工程比較
-->
---

<NavBar active="literature" />

<SectionTag>§ Literature review · training supervision in prior work</SectionTag>

# Prior reconstruction methods require full-field supervision

<div class="text-sm mt-1" style="color:#374151;">
Their architectures differ, but each method is trained against a dense reference field that is unavailable on a deployed rig.
</div>

<style>
.dns { width:100%; border-collapse:collapse; font-size:.90rem; margin-top:14px; }
.dns th { text-align:left; font-weight:700; color:#9CA3AF; font-size:.68rem; text-transform:uppercase; letter-spacing:.05em; padding:0 10px 6px; border-bottom:1px solid #D8D2E0; }
.dns th.key { color:#E97132; }
.dns td { padding:8px 10px; border-bottom:1px solid #F1EDF5; color:#374151; vertical-align:middle; line-height:1.22; }
.dns .model { display:block; color:#1F1B2E; font-size:.94rem; font-weight:600; }
.dns .cite { display:block; color:#9CA3AF; font-size:.76rem; line-height:1.18; margin-top:1px; white-space:normal; }
.dns .mechanism { font-weight:600; color:#1F1B2E; }
.dns .target { color:#E97132; font-weight:700; }
.dns .req { display:block; margin-top:2px; color:#E97132; font-size:.72rem; font-weight:700; letter-spacing:.03em; }
.dns-summary { margin-top:14px; padding-left:12px; border-left:2px solid #E97132; color:#374151; font-size:.90rem; line-height:1.3; }
</style>

<table class="dns">
<thead>
<tr>
<th style="width:25%;">Work</th>
<th style="width:25%;">Reconstruction mechanism</th>
<th style="width:20%;">Flow case</th>
<th style="width:30%;" class="key">Training target</th>
</tr>
</thead>
<tbody>
<tr>
<td><span class="model">SHRED</span><span class="cite">Williams et al. (2024)<br/>Proc. R. Soc. A</span></td>
<td class="mechanism">LSTM stack + shallow decoder</td>
<td>Isotropic (JHTDB), Re 2.3×10⁴</td>
<td><span class="target">Full state, ‖x − H(y)‖₂</span><br/><span class="req">FULL FIELD REQUIRED</span></td>
</tr>
<tr>
<td><span class="model">Senseiver</span><span class="cite">Santos et al. (2023)<br/>Nat. Mach. Intell.</span></td>
<td class="mechanism">Perceiver IO, cross-attention</td>
<td>Re not stated</td>
<td><span class="target">Dense observations for training</span><br/><span class="req">FULL FIELD REQUIRED</span></td>
</tr>
<tr>
<td><span class="model">FLRNet</span><span class="cite">Nguyen et al. (2024)<br/>arXiv</span></td>
<td class="mechanism">CNN-VAE + Fourier features + MLP</td>
<td>Cylinder, Re 300–10³</td>
<td><span class="target">Full field, VAE + perceptual loss</span><br/><span class="req">FULL FIELD REQUIRED</span></td>
</tr>
<tr>
<td><span class="model">FLRONet</span><span class="cite">Vo Dang &amp; Nguyen (2024)<br/>J. Comput. Inf. Sci. Eng.</span></td>
<td class="mechanism">DeepONet, FNO branch + MLP trunk</td>
<td>Cylinder, Re not stated</td>
<td><span class="target">Paired CFD fields</span><br/><span class="req">FULL FIELD REQUIRED</span></td>
</tr>
</tbody>
</table>

<div class="dns-summary"><b>Deployment gap:</b> a rig provides sensor histories, not the dense DNS/CFD field required to supervise these methods during training.</div>


<FooterLogos />

<!--
[學習式方法對著什麼擬合 · 1.5min]
• 論點只有一個：四篇全都對著 reference field 擬合
• 問「有無 DeepONet 對照」→ FLRONet；真正對照是內部 B0
⚠️⚠️ 不可說「Re 比所有文獻高」！SHRED 的 JHTDB Re=23,298 是我們 2.3 倍
　　 正確講法：「不用全場監督的同類裡最高」
⚠️ 四篇有三篇未報 Re
-->

---

<NavBar active="literature" />

<SectionTag>§ Literature review · how the sensor stream is used</SectionTag>

# How the sensor stream enters the model

<style>
.se { width: 100%; border-collapse: collapse; font-size: 0.90rem; margin-top: 14px; }
.se th { text-align: left; font-weight: 700; color: #9CA3AF; font-size: 0.68rem; text-transform: uppercase;
         letter-spacing: 0.05em; padding: 0 10px 6px 10px; border-bottom: 1px solid #D8D2E0; }
.se td { padding: 11px 10px; border-bottom: 1px solid #F1EDF5; color: #374151; vertical-align: top; line-height: 1.3; }
.se td.who { color: #1F1B2E; font-weight: 600; }
.se td.who span { display: block; font-weight: 400; color: #9CA3AF; font-size: 0.78em; margin-top: 1px; }
.se .no { color: #E97132; font-weight: 700; }
.se .yes { color: #0F2D52; font-weight: 700; }
.se .partial { color: #9CA3AF; font-weight: 700; }
</style>

<table class="se">
<thead>
<tr>
<th style="width: 27%;">Approach</th>
<th style="width: 25%;">Sensors as input</th>
<th style="width: 17%;">Query anywhere</th>
<th style="width: 15%;">Uneven clocks</th>
<th style="width: 16%;">PDE residual</th>
</tr>
</thead>
<tbody>
<tr>
<td class="who">Coordinate PINN <span>Raissi 2019, J. Comput. Phys.</span></td>
<td><span class="no">✗</span> scored by a loss term</td>
<td><span class="yes">✓</span></td>
<td><span class="no">✗</span></td>
<td><span class="yes">✓</span></td>
</tr>
<tr>
<td class="who">Operator networks <span>Lu 2021, Nat. Mach. Intell., Li 2021, ICLR</span></td>
<td><span class="no">✗</span> needs a dense grid</td>
<td><span class="yes">✓</span></td>
<td><span class="no">✗</span></td>
<td><span class="partial">on a grid</span></td>
</tr>
<tr>
<td class="who">Sensor-input networks <span>Williams 2024, Proc. R. Soc. A, Santos 2023, Nat. Mach. Intell.</span></td>
<td><span class="yes">✓</span></td>
<td><span class="partial">decoder-bound</span></td>
<td><span class="no">✗</span></td>
<td><span class="no">✗</span></td>
</tr>
<tr>
<td class="who">Continuous-time cells <span>Hasani 2022, Nat. Mach. Intell., Chen 2018, NeurIPS</span></td>
<td><span class="yes">✓</span></td>
<td><span class="no">✗</span> no spatial field</td>
<td><span class="yes">✓</span></td>
<td><span class="no">✗</span></td>
</tr>
</tbody>
</table>

<div class="mt-4" style="display:grid; grid-template-columns:max-content 1fr; column-gap:14px; align-items:baseline; font-size:0.90rem; border-left:2px solid #E97132; padding-left:12px;">
<span style="color:#9CA3AF; white-space:nowrap;">Gap</span><span style="color:#374151;">Every approach is missing at least one column. The capability needed here is <b>all four at once</b>.</span>
</div>

<FooterLogos />

<!--
[感測資料如何進入模型 · 1.5min]
• 四欄＝四種能力，每一列都缺至少一欄
• PINN 缺 sensor input／operator 要規則網格／sensor-input 無 PDE／CfC 無空間場
⚠️ 本段不放 PI-CON
-->

---

<NavBar active="motivation" />

<SectionTag>§ Motivation · operator vs. plain PINN</SectionTag>

# The key distinction is <span style="color:#7F1084;">where the measurements enter</span>

<style>
.opcmp { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:14px; }
.opcmp .panel { border:1px solid #E5E0EC; border-radius:10px; padding:13px 15px 12px; background:rgba(255,255,255,.82); }
.opcmp .panel.operator { border-color:#C9A6CC; background:#FAF2FB; }
.opcmp .head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:9px; }
.opcmp .name { font-size:.88rem; font-weight:700; color:#0F2D52; }
.opcmp .operator .name { color:#7F1084; }
.opcmp .tag { font-size:.68rem; color:#6B7280; }
.opcmp .flow { display:grid; grid-template-columns:1.12fr 24px 1fr 24px .9fr; align-items:center; gap:3px; margin-top:10px; }
.opcmp .node { min-height:58px; border:1px solid #D8DCE3; border-radius:7px; background:#fff; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; padding:6px; color:#1F1B2E; font-size:.74rem; line-height:1.28; }
.opcmp .node.fit { border-color:#E7B78F; background:#FFF7F1; color:#9A4B16; font-weight:700; }
.opcmp .node.map { border-color:#A96BAD; color:#7F1084; font-weight:700; }
.opcmp .arr { color:#9CA3AF; text-align:center; font-size:1rem; }
.opcmp .stage { margin-top:11px; color:#6B7280; font-size:.67rem; letter-spacing:.06em; text-transform:uppercase; font-weight:700; }
.opcmp .infer { display:grid; grid-template-columns:1fr 24px 1fr 24px .9fr; align-items:center; gap:3px; margin-top:5px; }
.opcmp .merge { display:grid; grid-template-columns:1fr 24px 1.1fr 24px .9fr; grid-template-rows:1fr 1fr; align-items:center; gap:6px 3px; margin-top:10px; }
.opcmp .merge .model { grid-column:3; grid-row:1 / 3; min-height:112px; }
.opcmp .merge .out { grid-column:5; grid-row:1 / 3; min-height:76px; }
.opcmp .merge .to-model { grid-column:2; }
.opcmp .merge .to-out { grid-column:4; grid-row:1 / 3; }
.opcmp .note { margin-top:11px; padding-top:8px; border-top:1px solid #E5E7EB; color:#4B5563; font-size:.76rem; line-height:1.35; }
</style>

<div class="opcmp">

<div class="panel">
  <div class="head"><span class="name">Plain PINN</span><span class="tag">measurements enter through optimization</span></div>
  <div class="stage">Training</div>
  <div class="flow">
    <div class="node">sensor values<br/>+ NS residual</div><div class="arr">→</div>
    <div class="node fit">optimize θ</div><div class="arr">→</div><div class="node">fitted weights</div>
  </div>
  <div class="stage">Inference</div>
  <div class="infer">
    <div class="node">query<br/>(x, y, t)</div><div class="arr">→</div>
    <div class="node">PINN Nθ<br/><span style="color:#6B7280;">coordinate model</span></div><div class="arr">→</div><div class="node">û(x, y, t)</div>
  </div>
  <div class="note">At inference, the sensor history is no longer an input; its information is stored indirectly in θ.</div>
</div>

<div class="panel operator">
  <div class="head"><span class="name">Operator formulation</span><span class="tag">measurements enter the forward pass</span></div>
  <div class="stage">Conditioned query</div>
  <div class="merge">
    <div class="node">sensor history<br/>s(·)</div><div class="arr to-model">→</div>
    <div class="node map model">operator Gθ[s]<br/><span style="font-weight:400;color:#6B7280;">measurement-conditioned<br/>coordinate model</span></div>
    <div class="arr to-out">→</div><div class="node out">û(x, y, t)</div>
    <div class="node">query<br/>(x, y, t)</div><div class="arr to-model">→</div>
  </div>
  <div class="note">The sensor record and query coordinate jointly determine each reconstructed value.</div>
</div>

</div>

<div class="mt-3 px-4 py-2 rounded text-sm leading-snug" style="background:rgba(127,16,132,.06); border-left:4px solid #7F1084;">
<b style="color:#7F1084;">Why this matters:</b> sensor history remains an explicit input at inference, while the field stays queryable at any <i>(x, y, t)</i>.
</div>

<FooterLogos />

<!--
[量測從哪裡進入 · 2min]
• 只問一句：量測從哪裡進入模型
• 左＝只進 loss，資訊烘進 θ／右＝sensor + query 同時輸入
⚠️ 不可說「新 flow 不需重訓」
-->

---

<NavBar active="motivation" />

<SectionTag>§ Motivation · the two operator families</SectionTag>

# What a neural operator learns

<div class="text-sm mt-1" style="color:#374151;">
A network maps a point to a value. An <b>operator</b> maps a whole input function to a whole output function, so one trained model serves new inputs without retraining.
</div>

<style>
.op { display: grid; grid-template-columns: 1fr 1fr; column-gap: 22px; margin-top: 14px; }
.op .hd { font-size: 1.42rem; font-weight: 700; letter-spacing: 0.01em; text-transform: none; margin-bottom: 6px; }
.op .hd span { font-size: 1rem; }
.op .note { font-size: 0.84rem; line-height: 1.45; color: #374151; margin-top: 8px; }
</style>

<div class="op">

<div>
<div class="hd" style="color:#0F2D52;">FNO <span style="font-weight:400; text-transform:none; letter-spacing:0; color:#9CA3AF;">Li 2021, ICLR</span></div>
<svg viewBox="0 0 400 150" style="width:100%;height:auto;">
  <rect x="6" y="55" width="62" height="40" rx="4" fill="#F4F6F9" stroke="#0F2D52" stroke-width="1.4"/>
  <text x="37.0" y="75.0" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:12px;font-weight:700;">a(x)</text>
  <line x1="68" y1="75" x2="86" y2="75" stroke="#9CA3AF" stroke-width="1.2"/>
  <path d="M92 75 L85 71.5 L85 78.5 Z" fill="#9CA3AF"/>
  <rect x="92" y="55" width="54" height="40" rx="4" fill="#FFF" stroke="#0F2D52" stroke-width="1.4"/>
  <text x="119.0" y="75.0" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:11px;font-weight:400;">FFT</text>
  <line x1="146" y1="75" x2="164" y2="75" stroke="#9CA3AF" stroke-width="1.2"/>
  <path d="M170 75 L163 71.5 L163 78.5 Z" fill="#9CA3AF"/>
  <rect x="170" y="55" width="54" height="40" rx="4" fill="#F4F6F9" stroke="#0F2D52" stroke-width="1.4"/>
  <text x="197.0" y="75.0" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:12px;font-weight:700;">× R</text>
  <line x1="224" y1="75" x2="242" y2="75" stroke="#9CA3AF" stroke-width="1.2"/>
  <path d="M248 75 L241 71.5 L241 78.5 Z" fill="#9CA3AF"/>
  <rect x="248" y="55" width="58" height="40" rx="4" fill="#FFF" stroke="#0F2D52" stroke-width="1.4"/>
  <text x="277.0" y="75.0" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:11px;font-weight:400;">iFFT</text>
  <line x1="306" y1="75" x2="324" y2="75" stroke="#9CA3AF" stroke-width="1.2"/>
  <path d="M330 75 L323 71.5 L323 78.5 Z" fill="#9CA3AF"/>
  <rect x="330" y="55" width="62" height="40" rx="4" fill="#F4F6F9" stroke="#0F2D52" stroke-width="1.4"/>
  <text x="361.0" y="75.0" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:12px;font-weight:700;">u(x)</text>
  <text x="197" y="40" text-anchor="middle" fill="#9CA3AF" style="font-size:10px;">keep low-k modes</text>
  <text x="199" y="122" text-anchor="middle" fill="#9CA3AF" style="font-size:10px;">input sampled on a regular grid</text>
</svg>
<div class="note">Convolves in Fourier space, so it sees the whole domain at once, but the input must arrive on a <b>regular grid</b>.</div>
</div>

<div>
<div class="hd" style="color:#7F1084;">DeepONet <span style="font-weight:400; text-transform:none; letter-spacing:0; color:#9CA3AF;">Lu 2021, Nat. Mach. Intell.</span></div>
<svg viewBox="0 0 400 150" style="width:100%;height:auto;">
  <rect x="6" y="18" width="86" height="34" rx="4" fill="#FAF2FB" stroke="#7F1084" stroke-width="1.4"/>
  <text x="49.0" y="35.0" text-anchor="middle" dominant-baseline="central" fill="#7F1084" style="font-size:11px;font-weight:700;">sensors</text>
  <line x1="92" y1="35" x2="110" y2="35" stroke="#C9A6CC" stroke-width="1.2"/>
  <path d="M116 35 L109 31.5 L109 38.5 Z" fill="#C9A6CC"/>
  <rect x="116" y="18" width="74" height="34" rx="4" fill="#FFF" stroke="#7F1084" stroke-width="1.4"/>
  <text x="153.0" y="35.0" text-anchor="middle" dominant-baseline="central" fill="#7F1084" style="font-size:11px;font-weight:400;">branch</text>
  <rect x="6" y="98" width="86" height="34" rx="4" fill="#FAF2FB" stroke="#7F1084" stroke-width="1.4"/>
  <text x="49.0" y="115.0" text-anchor="middle" dominant-baseline="central" fill="#7F1084" style="font-size:11px;font-weight:700;">query (x, t)</text>
  <line x1="92" y1="115" x2="110" y2="115" stroke="#C9A6CC" stroke-width="1.2"/>
  <path d="M116 115 L109 111.5 L109 118.5 Z" fill="#C9A6CC"/>
  <rect x="116" y="98" width="74" height="34" rx="4" fill="#FFF" stroke="#7F1084" stroke-width="1.4"/>
  <text x="153.0" y="115.0" text-anchor="middle" dominant-baseline="central" fill="#7F1084" style="font-size:11px;font-weight:400;">trunk</text>
  <path d="M190 35 C 222 35, 222 75, 244 75" stroke="#C9A6CC" stroke-width="1.2" fill="none"/>
  <path d="M190 115 C 222 115, 222 75, 244 75" stroke="#C9A6CC" stroke-width="1.2" fill="none"/>
  <circle cx="258" cy="75" r="15" fill="#FFF" stroke="#7F1084" stroke-width="1.4"/>
  <text x="258" y="75" text-anchor="middle" dominant-baseline="central" fill="#7F1084" style="font-size:15px;font-weight:700;">, </text>
  <line x1="273" y1="75" x2="294" y2="75" stroke="#C9A6CC" stroke-width="1.2"/>
  <path d="M300 75 L293 71.5 L293 78.5 Z" fill="#C9A6CC"/>
  <rect x="300" y="55" width="92" height="40" rx="4" fill="#FAF2FB" stroke="#7F1084" stroke-width="1.4"/>
  <text x="346.0" y="75.0" text-anchor="middle" dominant-baseline="central" fill="#7F1084" style="font-size:12px;font-weight:700;">u(x, t)</text>
  <text x="258" y="107" text-anchor="middle" fill="#9CA3AF" style="font-size:10px;">inner product</text>
  <text x="199" y="146" text-anchor="middle" fill="#9CA3AF" style="font-size:10px;">sensors may sit anywhere; query is continuous</text>
</svg>
<div class="note">Splits the map in two: a <b>branch</b> reads the input function, a <b>trunk</b> reads the query coordinate. Their inner product gives the value.</div>
</div>

</div>


<FooterLogos />

<!--
[兩種 operator · 1min]
• Operator 學「函數 → 函數」
• FNO 要規則網格／DeepONet branch-trunk 可散點、可任意查詢
• 我們是 100 散點 + 任意查詢 → 選 DeepONet
⚠️ 不提 CfC / cross-attn / AL（下一段）
⚠️ 圖為示意
-->

---

<NavBar active="motivation" />

<SectionTag>§ Motivation · problem formulation</SectionTag>

# A rig provides sparse histories and physics—not a full field

<style>
.ps { margin-top:18px; }
.ps .available { font-size:.73rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#6B7280; margin-bottom:7px; }
.ps .map { display:grid; grid-template-columns:1fr 44px 1fr 64px 1.08fr; align-items:center; }
.ps .node { min-height:176px; padding:17px 16px; border-radius:11px; background:rgba(255,255,255,.82); border:1px solid #E5E0EC; }
.ps .node .tag { font-size:.72rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:#6B7280; }
.ps .node .hero { font-size:1.78rem; color:#0F2D52; font-weight:700; line-height:1.1; margin:13px 0 9px; }
.ps .node .body { font-size:.86rem; color:#374151; line-height:1.36; }
.ps .symbol { text-align:center; font-size:1.75rem; color:#7F1084; font-weight:700; }
.ps .answer { background:#FAF2FB; border-color:#7F1084; }
.ps .answer .hero, .ps .answer .tag { color:#7F1084; }
.ps .constraint { margin-top:14px; padding:11px 15px; border-left:4px solid #E97132; background:#FEF6F1; color:#374151; font-size:.92rem; line-height:1.3; }
.ps .constraint b { color:#E97132; }
</style>

<div class="ps">
  <div class="available">Available during deployment</div>
  <div class="map">
    <div class="node"><div class="tag">Sparse measurements</div><div class="hero" style="color:#E97132;">K = 100 probes</div><div class="body">Each fixed location reports only its velocity history: <b>y<sub>k</sub>(t) = (u,v)</b>.</div></div>
    <div class="symbol">+</div>
    <div class="node"><div class="tag">Known physical prior</div><div class="hero">Navier–Stokes</div><div class="body">Momentum and continuity restrict which full fields are physically admissible.</div></div>
    <div class="symbol">→</div>
    <div class="node answer"><div class="tag">Required output</div><div class="hero">û(x,t)</div><div class="body">A continuous reconstruction that can be queried at any coordinate—not only on a fixed grid.</div></div>
  </div>
  <div class="constraint"><b>Unavailable:</b> no dense reference field may supervise training or enter inference.</div>
</div>

<FooterLogos />

<!--
[問題定式]
• 排在 literature 之後：三類方法都不符部署條件
-->

---

<NavBar active="motivation" />

<SectionTag>§ Motivation · resolution limit under a sparse sensor budget</SectionTag>

# Sensor-count scale and spectral conditioning at K = 100

<style>
.resolution { display:grid; grid-template-columns:34% 66%; gap:20px; align-items:stretch; margin-top:13px; }
.resolution .cardx { background:rgba(255,255,255,.78); border:1px solid #E5E0EC; border-radius:10px; padding:14px; }
.resolution .tiny { font-size:.72rem; letter-spacing:.07em; text-transform:uppercase; font-weight:700; color:#6B7280; }
.resolution .formula { font-size:1.12rem; color:#7F1084; font-weight:700; text-align:center; margin:9px 0 9px; padding:7px 6px; background:#FAF2FB; border-radius:7px; }
.resolution .copy { font-size:.85rem; color:#374151; line-height:1.4; }
.resolution .summary { display:flex; flex-direction:column; }
.resolution .summary .formula { font-size:1.04rem; margin:7px 0; padding:6px; }
.resolution .summary .copy { font-size:.80rem; line-height:1.31; }
.resolution .energy-callout { margin-top:auto; padding:12px 11px; background:#F4F6F9; border-left:3px solid #7F1084; border-radius:0 7px 7px 0; }
.resolution .energy-callout .value { color:#7F1084; font-size:1.55rem; font-weight:700; line-height:1; }
.resolution .energy-callout .label { margin-top:5px; color:#374151; font-size:.76rem; line-height:1.28; }
.resolution .plot img { width:100%; height:270px; object-fit:contain; display:block; }
</style>

<div class="resolution">
  <div class="cardx summary">
    <div class="tiny">Sensor budget: K = 100</div>
    <div class="copy" style="margin-bottom:6px;"><b>Wavenumber <span class="raw">k</span></b>: low = large structures; high = fine structures.</div>
    <div class="formula">k<sub>sensor</sub> ≡ √(K/π) = 5.64</div>
    <div class="copy"><b>Sensor-count scale:</b> match K points to πk² modes—not a recovery bound.</div>
    <div class="energy-callout">
      <div class="value">98.9%</div>
      <div class="label">of DNS kinetic energy lies below k<sub>sensor</sub>.</div>
    </div>
  </div>
  <div class="cardx plot">
    <div class="tiny">DNS spectrum and cumulative energy at <span class="raw">t</span> = 5</div>
    <img :src="'/images/nyquist_recoverability.png'" />
    <div class="copy" style="font-size:.77rem;">Dashed line: k<sub>sensor</sub> = √(K/π). Energy below the line is <b>available</b>, not automatically recoverable.</div>
  </div>
</div>

<FooterLogos />

<!--
[K=100 能買到什麼 · 2min]
• 650× underdetermined；CS bound 還差 50×
• 可行範圍＝low-band + physics prior 補 null-space
• 收尾：加 sensor，不是加大網路
-->

---

<NavBar active="objective" />

<SectionTag>§ Literature review · same-regime works</SectionTag>

# Same regime: sensors + PDE, no reference field

<style>
.hh { width: 100%; border-collapse: collapse; font-size: 1.02rem; margin-top: 16px; }
.hh th { text-align: left; font-weight: 700; color: #9CA3AF; font-size: 0.72rem; text-transform: uppercase;
         letter-spacing: 0.05em; padding: 0 12px 8px 12px; border-bottom: 1px solid #D8D2E0; }
.hh td { padding: 13px 12px; border-bottom: 1px solid #F1EDF5; color: #374151; }
.hh .who { color: #1F1B2E; font-weight: 600; white-space: nowrap; }
.hh .who span { display: block; font-weight: 400; color: #9CA3AF; font-size: 0.72em; margin-top: 2px; }
.hh tr.ours td { background: #F7EDF8; border-bottom: none; color: #7F1084; font-weight: 700; }
</style>

<table class="hh">
<thead>
<tr>
<th style="width: 30%;">Work</th>
<th style="width: 14%;">Re</th>
<th style="width: 16%;">Probes</th>
<th style="width: 20%;">Sensors as input</th>
<th style="width: 20%;">Readout</th>
</tr>
</thead>
<tbody>
<tr>
<td class="who">Mo &amp; Magri 2025 <span>PC-DualConvNet · <i>Physical Review Fluids</i></span></td>
<td>34</td>
<td>230</td>
<td>✓</td>
<td>128² fixed mesh</td>
</tr>
<tr>
<td class="who">Kelshaw et al. 2022 <span>VDSR CNN · NeurIPS ML4PS Workshop</span></td>
<td>34</td>
<td>100</td>
<td>✓</td>
<td>150² fixed mesh</td>
</tr>
<tr>
<td class="who">Parfenyev et al. 2024 <span>coordinate-MLP PINN · <i>JETP Letters</i></span></td>
<td>1.3×10³</td>
<td>150 / snapshot <span style="display:block; color:#9CA3AF; font-size:0.72em;">3×10⁴ scattered (r, t)</span></td>
<td>✗ loss term only</td>
<td>query-anywhere</td>
</tr>
<tr class="ours">
<td class="who" style="color:#7F1084;">PI-CON (ours) <span style="color:#B98ABD;">DeepONet + CfC</span></td>
<td>10⁴</td>
<td>100</td>
<td>✓</td>
<td>query-anywhere</td>
</tr>
</tbody>
</table>

<div class="mt-3" style="display:grid; grid-template-columns:max-content 1fr; column-gap:14px; align-items:baseline; font-size:0.88rem; border-left:2px solid #7F1084; padding-left:12px;">
<span style="color:#9CA3AF;">Gap</span><span style="color:#374151;">No surveyed work combines <b style="color:#7F1084;">query-anywhere readout</b>, <b style="color:#7F1084;">sensors as input</b>, and <b style="color:#7F1084;">Re = 10⁴</b>.</span>
</div>

<FooterLogos />

<!--
[同 regime 的三篇 · 2min]
• 開場必講：同 regime survey **只找到三篇**
• ① 最近一篇 Re 仍低 7.7×　② Parfenyev 量測比我們多（150/snapshot）　③ 無人同時做到三件事
⚠️ 不可說「Parfenyev 不需 sensor」
⚠️ 問 FLRONet → 納入標準是監督訊號；它架構最接近但無法同 regime 對打
⚠️ 不宣稱勝過 Mo & Magri（指標不同、Re 差 300×）
-->

---

<NavBar active="objective" />

<SectionTag>§ Objective</SectionTag>

# Build a queryable flow field—and map what controls its fidelity

<style>
.objective-target { margin-top:14px; padding:15px 22px; border:1px solid #C9A6CC; border-radius:12px;
                    background:linear-gradient(100deg,#FAF2FB,rgba(255,255,255,.86)); text-align:center; }
.objective-target .label { color:#7F1084; font-size:.66rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
.objective-target .flow { margin-top:3px; color:#1F1B2E; font-size:1.35rem; font-weight:700; line-height:1.25; }
.objective-target .arrow { color:#7F1084; padding:0 10px; }
.objective-target .sub { margin-top:5px; color:#6B7280; font-size:.75rem; }
.objective-work { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:15px; }
.objective-work .verb { color:#7F1084; font-size:1.05rem; font-weight:700; line-height:1.2; }
.objective-work .body { margin-top:6px; color:#374151; font-size:.82rem; line-height:1.36; }
.objective-work .detail { margin-top:10px; padding-top:8px; border-top:1px solid #EFEAF2; color:#6B7280; font-size:.74rem; line-height:1.32; }
</style>

<div class="objective-target">
<div class="label">Research target</div>
<div class="flow"><span>sparse (u, v) sensor histories</span><span class="arrow">→</span><span>2-D velocity field at any (x, t)</span></div>
<div class="sub">Training constraint: Navier–Stokes residual &nbsp;·&nbsp; <b style="color:#7F1084;">no dense DNS field</b>.</div>
</div>

<div class="objective-work">

<Card>
<LabelTiny style="color:#7F1084;">O1</LabelTiny>
<div class="verb">Build the field reconstructor</div>
<div class="body">Train on sensor histories with a Navier–Stokes residual constraint.</div>
<div class="detail">Read out a continuous field at arbitrary space–time queries.</div>
</Card>

<Card>
<LabelTiny style="color:#7F1084;">O2</LabelTiny>
<div class="verb">Map the resolution scale</div>
<div class="body">Vary the number of sensors and examine the flow detail retained.</div>
<div class="detail">Separate large-scale recovery from missing fine-scale information.</div>
</Card>

<Card>
<LabelTiny style="color:#7F1084;">O3</LabelTiny>
<div class="verb">Test reconstruction reliability</div>
<div class="body">Vary sensor placement and perturb observations with measurement noise.</div>
<div class="detail">Identify which sensing conditions most affect reconstruction quality.</div>
</Card>

</div>

<FooterLogos />

<!--
[研究目標 · 1.5min]
• 先看中央：研究目標是從稀疏速度量測與 NS 殘差，得到任意 (x,t) 可查詢的 2D 速度場；訓練不使用 DNS 全場
• O1 建立重建器；O2 將感測器數量和可保留的空間細節連結；O3 檢驗佈點與雜訊下的可靠性
• 橋接：「後面結果依序回答：能否重建、哪些尺度受限、以及量測條件的影響。」
-->

---

<NavBar active="method" />

<SectionTag>§ Application case · the Kolmogorov flow</SectionTag>

# Reynolds number follows the benchmark convention

<style>
.ndflow { display:grid; grid-template-columns:43% 57%; gap:22px; margin-top:8px; align-items:start; }
.ndflow .flow-gif { width:100%; height:290px; object-fit:contain; border:1px solid #E5E0EC; border-radius:10px; background:rgba(255,255,255,.76); }
.ndflow .caption { margin-top:4px; text-align:center; color:#6B7280; font-size:.68rem; }
.ndflow .right { display:flex; flex-direction:column; gap:8px; }
.ndflow .source { padding:5px 10px; border-left:3px solid #7F1084; color:#374151; font-size:.70rem; line-height:1.28; }
.ndflow .source b { color:#7F1084; }
.ndflow .box { border:1px solid #E5E0EC; border-radius:10px; background:rgba(255,255,255,.80); padding:8px 12px; }
.ndflow .label { font-size:.61rem; letter-spacing:.07em; text-transform:uppercase; color:#6B7280; font-weight:700; margin-bottom:1px; }
.ndflow .ns { border-color:#C9A6CC; background:#FAF2FB; }
.ndflow .math { text-align:center; }
.ndflow .ns .math :deep(.katex-display) { margin:.25em 0; color:#0F2D52; font-size:.82em; }
.ndflow .definition { border-color:#7F1084; background:linear-gradient(110deg, #FAF2FB, rgba(255,255,255,.90)); }
.ndflow .definition .math :deep(.katex-display) { margin:.10em 0 .15em; color:#7F1084; font-size:1.02em; }
.ndflow .params { text-align:center; color:#6B7280; font-size:.66rem; }
.ndflow .takeaway { margin:0; text-align:center; color:#374151; font-size:.65rem; line-height:1.22; }
</style>

<div class="ndflow">
<div>
<img :src="'/images/kolmogorov_dns_vorticity_anim.gif'" class="flow-gif" />
<div class="caption">DNS vorticity · periodic unit square</div>
</div>
<div class="right">
<div class="source"><b>Literature anchor.</b> Wang et al. (2025) formulate the same unit-square Kolmogorov benchmark with a viscous coefficient of <span style="white-space:nowrap;">1 / Re</span>.</div>
<div class="box ns">
<div class="label" style="color:#7F1084;">Nondimensional Navier–Stokes convention</div>

<div class="math">

$$\partial_t \mathbf{u} + (\mathbf{u}\!\cdot\!\nabla)\mathbf{u} = -\nabla p + \frac{1}{\mathrm{Re}}\nabla^2\mathbf{u} + \mathbf{f}, \qquad \nabla\!\cdot\!\mathbf{u}=0$$

</div>
<div class="params">Ω = [0,1]² &nbsp; · &nbsp; f = (0.1 sin 4πy, 0)</div>
</div>
<div class="box definition">
<div class="label">Definition used in this work</div>

<div class="math">

$$\boxed{\mathrm{Re}\equiv\frac{1}{\nu}} \qquad \nu=10^{-4}\;\Longrightarrow\;\boxed{\mathrm{Re}=10^4}$$

</div>
<p class="takeaway">Thus, the coefficient of ∇²u is 10⁻⁴—not a separately estimated flow-scale Reynolds number.</p>
</div>
</div>
</div>

<FooterLogos />

<!--
[Kolmogorov flow · 1.5min]
• Re 的定義直接跟隨 Wang et al. 的無因次 NS：黏性項係數為 1/Re
• 因此本設定 ν = 10⁻⁴ 時，Re = 10⁴；不另引入 DNS-derived characteristic Re
⚠️ 全簡報使用此無因次 convention；不要改寫成未定義的 U L / ν
• 問「多湍流」→ 斜率 −4.61、無慣性 range
-->

---

<NavBar active="method" />

<SectionTag>§ Application case · numerical setup</SectionTag>

# How the reference data are generated

<style>
.st { display: grid; grid-template-columns: max-content 1fr; column-gap: 12px; row-gap: 2px;
      align-items: baseline; font-size: 0.73rem; margin-top: 2px; }
.st .k { color: #6B7280; white-space: nowrap; }
.st .v { color: #1F1B2E; }
.vf { width: 100%; border-collapse: collapse; font-size: 0.72rem; margin-top: 3px; }
.vf td { padding: 1.5px 0; vertical-align: baseline; }
.vf td.c { color: #6B7280; padding-right: 10px; white-space: nowrap; }
.vf td.m { text-align: right; font-variant-numeric: tabular-nums; color: #1F1B2E; font-weight: 600;
           padding-right: 6px; white-space: nowrap; }
.vf td.j { width: 1%; white-space: nowrap; }
.ok { color: #0F2D52; font-weight: 700; }
.warn { color: #E97132; font-weight: 700; }
.cap2 { font-size: 0.70rem; color: #6B7280; line-height: 1.35; margin-top: 4px; }
</style>

<div class="grid gap-3 mt-2" style="grid-template-columns: 1.06fr 0.94fr;">

<div class="space-y-1">

<Card>
<LabelTiny>DNS solver</LabelTiny>
<div class="st">
<span class="k">Scheme</span><span class="v">pseudo-spectral, 2/3 dealiasing, ETDRK4, fp64</span>
<span class="k">Grid</span><span class="v">run <b style="color:#7F1084;">1024²</b>, stored <b>256²</b></span>
<span class="k">Time step</span><span class="v">Δt = 2.5×10⁻⁴ <span style="color:#9CA3AF;">— solver integration step</span></span>
<span class="k">Simulation time</span><span class="v"><b>t ∈ [0, 5]</b> <span style="color:#9CA3AF;">— T = 5</span></span>
<span class="k">Sampling</span><span class="v">Δt<sub>s</sub> = 0.025 <span style="color:#9CA3AF;">— every 100 steps</span></span>
<span class="k">Snapshots</span><span class="v">N<sub>t</sub> = 201 <span style="color:#9CA3AF;">— including t = 0</span></span>
</div>
</Card>

<Card>
<LabelTiny>What was verified</LabelTiny>
<table class="vf">
<tbody>
<tr><td class="c">Continuity residual <span style="color:#9CA3AF;">(run grid)</span></td><td class="m">‖∇·u‖<sub>∞</sub> ≲ 10⁻¹²</td><td class="j"><span class="ok">round-off</span></td></tr>
<tr><td class="c">Spatial resolution <span style="color:#9CA3AF;">[Pope 2000]</span></td><td class="m">k<sub>max</sub><span class="raw">η</span> = 7.61 / 1.91 <span style="color:#9CA3AF;">≥ 1.5</span></td><td class="j"><span class="ok">resolved</span></td></tr>
</tbody>
</table>
<div class="cap2"><b style="color:#0F2D52;">Scope:</b> one DNS realisation, not a statistically converged ensemble.</div>
</Card>

</div>

<div>
<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<img :src="'/images/sensor_distribution_kolmogorov_K100.png'" style="width: 100%; max-height: 246px; object-fit: contain;" />
<div class="cap2"><b style="color:#7F1084;">K = 100</b> fixed sensors; observed quantities: (u, v).</div>
</Card>
</div>

</div>

<FooterLogos />

<!--
[數值設定 · 1.5min]
• 投影片只留兩個主 gate：continuity residual ≲1e-12（round-off）與 Pope resolution 7.61/1.91
⚠️ div 是 solver run grid。儲存場非無散度（實測 8.7e-2）→ 問就答 FD floor 1.04 %
⚠️ T/t_eddy 未達標 → 講「單一 realisation」，不可說統計收斂
-->

---

<NavBar active="method" />

<SectionTag>§ Application case · DNS-free sensor placement</SectionTag>

# How sensor locations are generated without DNS

<style>
.lesst { display:grid; grid-template-columns:max-content 1fr; column-gap:12px; row-gap:2px;
         align-items:baseline; font-size:.73rem; margin-top:3px; }
.lesst .k { color:#6B7280; white-space:nowrap; }
.lesst .v { color:#1F1B2E; }
.lesvf { width:100%; border-collapse:collapse; font-size:.71rem; margin-top:3px; }
.lesvf td { padding:2px 0; vertical-align:baseline; }
.lesvf td.c { color:#6B7280; padding-right:10px; white-space:nowrap; }
.lesvf td.m { text-align:right; font-variant-numeric:tabular-nums; color:#1F1B2E; font-weight:600;
              padding-right:7px; white-space:nowrap; }
.lesvf td.j { width:1%; white-space:nowrap; }
.lescap { font-size:.69rem; color:#6B7280; line-height:1.35; margin-top:4px; }
.lespipe { margin-top:5px; padding-top:5px; border-top:1px solid #E5E0EC; text-align:center;
           color:#374151; font-size:.71rem; }
</style>

<div class="grid gap-3 mt-2" style="grid-template-columns:1.06fr .94fr;">

<div class="space-y-1">

<Card>
<LabelTiny>LES solver for placement</LabelTiny>
<div class="lesst">
<span class="k">Purpose</span><span class="v"><b style="color:#7F1084;">to decide where the sensors go</b></span>
<span class="k">Equation</span><span class="v">filtered NS with SGS stress and linear friction</span>
<span class="k">Solver</span><span class="v">pseudo-spectral, 2/3 dealiasing, RK2 Heun, fp64</span>
<span class="k">Grid / horizon</span><span class="v"><b>N = 256</b>, T<sub>end</sub> = 50</span>
<span class="k">Closure</span><span class="v">spectral hyperviscosity <span style="color:#9CA3AF;">— order p = 2, used alone</span></span>
<span class="k">Friction</span><span class="v"><b style="color:#E97132;">r = 2.86×10⁻²</b> <span style="color:#9CA3AF;">— absent from DNS</span></span>
<span class="k">Cost</span><span class="v">approximately <b>1/16 of DNS</b></span>
</div>
</Card>

<Card>
<LabelTiny>What was verified</LabelTiny>
<table class="lesvf">
<tbody>
<tr><td class="c">Continuity residual</td><td class="m">‖∇·u‖<sub>max</sub> = 2.29×10⁻¹³</td><td class="j"><span class="ok">round-off</span></td></tr>
<tr><td class="c">Alias control</td><td class="m">tail decay = 5.14×10³²</td><td class="j"><span class="ok">verified</span></td></tr>
</tbody>
</table>
<div class="lescap"><b style="color:#0F2D52;">Placement scope:</b> T<sub>end</sub>/<span class="raw">τ</span><sub>int</sub> = 4.9 &lt; 10; no claim of statistically converged LES data.</div>
</Card>

</div>

<div>
<Card style="padding-top:.6rem; padding-bottom:.6rem;">
<img :src="'/images/les_T50_vorticity_with_sensors.png'" style="width:100%; max-height:250px; object-fit:contain;" />
<div class="lespipe"><b>LES large-scale field</b> <span style="color:#C9C6D0;">→</span> QR-pivot <span style="color:#C9C6D0;">→</span> <b style="color:#7F1084;">K = 100 fixed locations</b></div>
<div class="lescap" style="color:#374151; margin-top:6px; padding-top:6px; border-top:1px solid #E5E0EC;">
<b style="color:#7F1084;">The LES gives us 100 coordinates. That is all.</b> It is never training data — the network is trained on sensor values and the NS residual.
</div>
</Card>
</div>

</div>

<FooterLogos />

<!--
[LES 佈點 · 1.5min]
• **LES 只決定 sensor 放在哪裡，不是訓練資料**（教授 2026-07-20 指定要講明）
  一句話版本：「LES 唯一的產出是 100 個座標。之後訓練看到的是那些點上的量測值加 NS residual，
  LES 場本身從頭到尾沒有進過網路。」
• 數值面達標（div 2.29×10⁻¹³、無 aliasing），但統計窗未建立（4.9 < 10）—— 這不影響佈點用途
⚠️ closure 是 hyperviscosity 單獨用；Bardina 只在 low-fidelity 變體
⚠️ 不可宣稱 LES 統計收斂。佈點品質的證據在下游：LES 5.71 % vs DNS-oracle 4.68 %
-->

---

<NavBar active="method" />

<SectionTag>§ Application case · which points carry the most information</SectionTag>

# Picking the K = 100 locations

<style>
.qp-flow { display:grid; grid-template-columns:175px 185px 1fr 175px; gap:16px; align-items:center; margin-top:14px; }
.qp-node { min-width:0; position:relative; }
.qp-node:not(:last-child)::after { content:"→"; position:absolute; right:-15px; top:64px; color:#C9C6D0; font-size:1.35rem; font-weight:400; }
.qp-no { font-size:.66rem; font-weight:700; letter-spacing:.08em; color:#E97132; text-transform:uppercase; }
.qp-node:last-child .qp-no { color:#7F1084; }
.qp-title { font-size:1.0rem; font-weight:700; color:#1F1B2E; margin-top:3px; line-height:1.2; }
.qp-caption { font-size:.73rem; color:#6B7280; line-height:1.28; margin-top:6px; }
.les-thumb { width:108px; height:108px; object-fit:cover; border-radius:7px; border:1px solid #E5E0EC; margin-top:10px; }
.feat-stack { margin-top:10px; display:grid; gap:3px; width:142px; }
.feat-stack span { display:block; border-left:3px solid #E97132; background:#F7F4F8; padding:3px 7px; color:#374151; font-size:.73rem; line-height:1.05; }
.feat-stack .more { color:#7F1084; font-weight:700; border-left-color:#7F1084; }
.feature-note { color:#7F1084; font-size:.72rem; margin-top:5px; }
.matrix-wrap { margin-top:10px; display:grid; grid-template-columns:28px 1fr; grid-template-rows:auto 1fr; column-gap:5px; align-items:center; }
.matrix-top { grid-column:2; display:flex; justify-content:space-between; align-items:center; color:#6B7280; font-size:.62rem; line-height:1; margin-bottom:4px; }
.matrix-side { grid-row:2; color:#6B7280; font-size:.59rem; line-height:1.05; text-align:center; writing-mode:vertical-rl; transform:rotate(180deg); }
.matrix { grid-column:2; display:grid; grid-template-columns:repeat(12, 1fr); gap:2px; padding:5px; border:1px solid #D8D2E0; border-radius:4px; background:#FAF9FB; }
.matrix i { display:block; height:8px; border-radius:1px; background:#D8D2E0; }
.matrix i.pick { background:#7F1084; }
.matrix-label { margin-top:4px; color:#374151; font-size:.72rem; text-align:center; }
.grid-map { position:relative; width:112px; height:112px; margin-top:10px; border:1px solid #C9C6D0; border-radius:5px; background:linear-gradient(90deg, transparent 24%, #E5E0EC 25%, transparent 26%, transparent 49%, #E5E0EC 50%, transparent 51%, transparent 74%, #E5E0EC 75%, transparent 76%), linear-gradient(0deg, transparent 24%, #E5E0EC 25%, transparent 26%, transparent 49%, #E5E0EC 50%, transparent 51%, transparent 74%, #E5E0EC 75%, transparent 76%); }
.grid-map i { position:absolute; width:9px; height:9px; border-radius:50%; background:#7F1084; box-shadow:0 0 0 2px #fff; }
.grid-map i:nth-child(1) { left:14px; top:17px; }.grid-map i:nth-child(2) { left:49px; top:12px; }.grid-map i:nth-child(3) { left:83px; top:25px; }.grid-map i:nth-child(4) { left:29px; top:48px; }.grid-map i:nth-child(5) { left:69px; top:57px; }.grid-map i:nth-child(6) { left:94px; top:79px; }.grid-map i:nth-child(7) { left:12px; top:85px; }.grid-map i:nth-child(8) { left:51px; top:92px; }
.rule { margin-top:13px; padding:12px 17px; border-radius:8px; background:rgba(127,16,132,0.06); border-left:4px solid #7F1084; font-size:.96rem; color:#374151; line-height:1.42; }
.rule p { margin:0; }
</style>

<div class="qp-flow">
  <div class="qp-node">
    <div class="qp-no">Step 1</div>
    <div class="qp-title">Run the LES</div>
    <img class="les-thumb" :src="'/images/les_vorticity_thumb.png'" />
    <div class="qp-caption">large-scale flow over <i>T</i> snapshots</div>
  </div>

  <div class="qp-node">
    <div class="qp-no">Step 2</div>
    <div class="qp-title">Stack five signals</div>
    <div class="feat-stack">
      <span><i>u</i>, <i>v</i></span><span><i>ω</i></span><span>|∇<i>u</i>|, |∇<i>v</i>|</span><span class="more">at every time</span>
    </div>
    <div class="feature-note">each channel → unit variance</div>
  </div>

  <div class="qp-node">
    <div class="qp-no">Step 3</div>
    <div class="qp-title">Build the feature matrix</div>
    <div class="matrix-wrap">
      <div class="matrix-top"><span>grid point 1</span><span>…</span><span>grid point <i>N</i>²</span></div>
      <div class="matrix-side">5 features × <i>T</i> times</div>
      <div class="matrix">
        <i></i><i></i><i class="pick"></i><i></i><i></i><i></i><i></i><i class="pick"></i><i></i><i></i><i></i><i></i>
        <i></i><i></i><i class="pick"></i><i></i><i></i><i></i><i></i><i class="pick"></i><i></i><i></i><i></i><i></i>
        <i></i><i></i><i class="pick"></i><i></i><i></i><i></i><i></i><i class="pick"></i><i></i><i></i><i></i><i></i>
        <i></i><i></i><i class="pick"></i><i></i><i></i><i></i><i></i><i class="pick"></i><i></i><i></i><i></i><i></i>
        <i></i><i></i><i class="pick"></i><i></i><i></i><i></i><i></i><i class="pick"></i><i></i><i></i><i></i><i></i>
      </div>
    </div>
    <div class="matrix-label"><i>A</i> ∈ ℝ<sup>5<i>T</i> × <i>N</i>²</sup> · one full column = one location</div>
  </div>

  <div class="qp-node">
    <div class="qp-no">Step 4</div>
    <div class="qp-title">Pivoted QR → locations</div>
    <div class="grid-map"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
    <div class="qp-caption"><b style="color:#7F1084;">K = 100</b> selected columns become fixed sensor coordinates</div>
  </div>
</div>

<div class="rule">

QR keeps the next **least-explained full column** — a location whose <i>whole signal history</i> adds new information beyond the points already selected.

</div>

<div class="foot mt-2">Gradients make shear-layer points distinguishable from smooth patches. The placement uses LES only; no DNS field is read.</div>

<FooterLogos />

<!--
[QR 佈點 · 1.5min]
• 講法：「不是隨便挑，也不是挑速度最大的點 —— 是挑**彼此最不重複**的點」
• 五個場各除以自身 std（否則梯度量級會壓過速度）
• 中央矩陣是示意：每個完整 column = 一個格點的 5T 維 signal history；紫色為 QR 選中的整欄
• 右邊小座標圖表示欄索引回到物理空間後，成為 K=100 個固定 sensor coordinates
• 為何帶梯度：剪切層上的點變化大、資訊多；平滑區的點彼此高度重複
⚠️ 這裡**沒有 POD／SVD／秩截斷** —— 對 raw feature matrix 直接做 column-pivoted QR。
   不可說成「取 leading modes」（CLAUDE.md 明列此禁項）
⚠️ 特徵清單核對 scripts/generate_sensors_qrpivot_from_les.py:93 的 features_used
⚠️ 行內公式要編譯，該 <div> 內容前後必須留空行（markdown-it 不處理 raw HTML 區塊內的 $）
-->

---

<NavBar active="method" />

<SectionTag>§ Methodology · the architecture</SectionTag>

# PI-CON: a CfC–DeepONet hybrid

<style>
.lg { display: flex; gap: 20px; align-items: center; font-size: 0.76rem; color: #374151; margin-top: 3px; }
.lg .sw { display: inline-block; width: 13px; height: 13px; border-radius: 3px; margin-right: 6px;
          vertical-align: -2px; }
.archviz { position:relative; }
</style>

<div class="lg">
<span><span class="sw" style="background:#0F2D52;"></span>Inherited DeepONet backbone</span>
<span><span class="sw" style="background:#D97757;"></span><b>Added in this work</b></span>
<span><span class="sw" style="background:#F7FAFD; border:1px dashed #0F2D52;"></span>DeepONet basis tensors</span>
<span><span class="sw" style="background:#FAF2FB; border:1px solid #7F1084;"></span>Training loop</span>
</div>

<div class="bg-gray-50 border border-gray-200 rounded-lg p-2 mt-2">

<div class="archviz">

<svg viewBox="0 0 1200 350" style="width:100%;height:auto;display:block;" aria-label="PI-CON architecture and training loop">
  <defs>
    <marker id="arch-forward-arrow" markerWidth="7" markerHeight="7" refX="6.2" refY="3.5" orient="auto">
      <path d="M0,0 L7,3.5 L0,7 Z" fill="#4B5563"/>
    </marker>
    <marker id="arch-train-arrow" markerWidth="7" markerHeight="7" refX="6.2" refY="3.5" orient="auto">
      <path d="M0,0 L7,3.5 L0,7 Z" fill="#7F1084"/>
    </marker>
  </defs>

  <!-- Forward paths: all use one shared arrow definition and coordinate system. -->
  <g fill="none" stroke="#4B5563" stroke-width="1.5" marker-end="url(#arch-forward-arrow)">
    <path d="M150 58 H190"/>
    <path d="M350 58 H390"/>
    <path d="M530 58 H620"/>
    <path d="M500 90 C500 118 492 128 486 147"/>
    <path d="M170 200 H190"/>
    <path d="M365 195 H420"/>
    <path d="M520 195 H620"/>
    <path d="M775 58 C820 58 820 116 842 128"/>
    <path d="M775 198 C810 198 820 180 842 170"/>
    <path d="M920 150 H990"/>
  </g>

  <!-- Query path. -->
  <rect x="20" y="25" width="130" height="65" fill="#F4F1FF" stroke="#9B7CF8" stroke-width="1.3"/>
  <text x="85" y="53" text-anchor="middle" fill="#1F1B2E" style="font-size:14px;font-weight:600;">Queries x,t</text>
  <text x="85" y="75" text-anchor="middle" fill="#6B7280" style="font-size:11.5px;">Nq × 4</text>

  <rect x="190" y="25" width="160" height="65" fill="#D97757" stroke="#D97757" stroke-width="1.3"/>
  <text x="270" y="52" text-anchor="middle" fill="#FFF" style="font-size:15px;font-weight:700;">Fourier embed</text>
  <text x="270" y="75" text-anchor="middle" fill="#FFEDE4" style="font-size:11.5px;font-weight:600;">Nq × 128</text>

  <rect x="390" y="25" width="140" height="65" fill="#0F2D52" stroke="#0F2D52" stroke-width="1.3"/>
  <text x="460" y="52" text-anchor="middle" fill="#FFF" style="font-size:15px;font-weight:700;">MLP trunk</text>
  <text x="460" y="75" text-anchor="middle" fill="#CFE0F2" style="font-size:11.5px;font-weight:600;">Nq × 256</text>

  <rect x="620" y="25" width="155" height="65" fill="#F7FAFD" stroke="#0F2D52" stroke-width="1.3" stroke-dasharray="5 4"/>
  <text x="697.5" y="52" text-anchor="middle" fill="#0F2D52" style="font-size:14px;font-weight:600;">trunk_basis</text>
  <text x="697.5" y="75" text-anchor="middle" fill="#536B86" style="font-size:11.5px;font-weight:600;">Nq × 3 × 256</text>

  <!-- Sensor path. -->
  <rect x="20" y="165" width="150" height="70" fill="#F4F1FF" stroke="#9B7CF8" stroke-width="1.3"/>
  <text x="95" y="194" text-anchor="middle" fill="#1F1B2E" style="font-size:14px;font-weight:600;">K=100 sensors</text>
  <text x="95" y="217" text-anchor="middle" fill="#6B7280" style="font-size:11.5px;">201 × 100 × 2</text>

  <rect x="190" y="155" width="175" height="80" fill="#D97757" stroke="#D97757" stroke-width="1.3"/>
  <text x="277.5" y="184" text-anchor="middle" fill="#FFF" style="font-size:14px;font-weight:700;">CfC branch</text>
  <text x="277.5" y="204" text-anchor="middle" fill="#FFF" style="font-size:14px;font-weight:700;">continuous-time</text>
  <text x="277.5" y="224" text-anchor="middle" fill="#FFEDE4" style="font-size:11.5px;font-weight:600;">201 × 100 × 256</text>

  <circle cx="470" cy="195" r="50" fill="#D97757" stroke="#D97757" stroke-width="1.3"/>
  <text x="470" y="181" text-anchor="middle" fill="#FFF" style="font-size:14px;font-weight:700;">Cross-Attn</text>
  <text x="470" y="201" text-anchor="middle" fill="#FFF" style="font-size:14px;font-weight:700;">+ dist. bias</text>
  <text x="470" y="223" text-anchor="middle" fill="#FFEDE4" style="font-size:10.5px;font-weight:600;">Nq × 100 × 256</text>

  <rect x="620" y="165" width="155" height="65" fill="#F7FAFD" stroke="#0F2D52" stroke-width="1.3" stroke-dasharray="5 4"/>
  <text x="697.5" y="192" text-anchor="middle" fill="#0F2D52" style="font-size:14px;font-weight:600;">branch_basis</text>
  <text x="697.5" y="215" text-anchor="middle" fill="#536B86" style="font-size:11.5px;font-weight:600;">Nq × 3 × 256</text>

  <!-- DeepONet readout. -->
  <path d="M842 120 H898 L920 150 L898 180 H842 L820 150 Z" fill="#0F2D52" stroke="#0F2D52" stroke-width="1.3"/>
  <text x="870" y="144" text-anchor="middle" fill="#FFF" style="font-size:14px;font-weight:700;">Inner</text>
  <text x="870" y="163" text-anchor="middle" fill="#FFF" style="font-size:14px;font-weight:700;">product</text>

  <rect x="990" y="120" width="120" height="60" fill="#F4F1FF" stroke="#9B7CF8" stroke-width="1.3"/>
  <text x="1050" y="145" text-anchor="middle" fill="#1F1B2E" style="font-size:14px;font-weight:600;">u, v, p</text>
  <text x="1050" y="166" text-anchor="middle" fill="#6B7280" style="font-size:11.5px;">Nq × 3</text>

  <!-- Training loop: output → loss → optimizer → the two trainable paths. -->
  <g fill="none" stroke="#7F1084" stroke-width="1.8" marker-end="url(#arch-train-arrow)">
    <path d="M1050 180 V302 H1042"/>
    <path d="M820 302 H792"/>
    <path d="M570 302 H520 L280 302 V236"/>
    <path d="M520 302 H560 V108 H420 V91"/>
  </g>

  <rect x="820" y="279" width="220" height="46" rx="7" fill="#FEF6F1" stroke="#E9A97E" stroke-width="1.3"/>
  <text x="930" y="297" text-anchor="middle" fill="#7F1084" style="font-size:12.5px;font-weight:700;">LOSS</text>
  <text x="930" y="315" text-anchor="middle" fill="#374151" style="font-size:11px;">sensor MSE + NS residual</text>

  <rect x="570" y="279" width="220" height="60" rx="7" fill="#FAF2FB" stroke="#C9A6CC" stroke-width="1.3"/>
  <text x="680" y="296" text-anchor="middle" fill="#7F1084" style="font-size:12.5px;font-weight:700;">OPTIMIZER</text>
  <text x="680" y="313" text-anchor="middle" fill="#374151" style="font-size:11px;">SOAP + Schedule-Free</text>
  <text x="680" y="330" text-anchor="middle" fill="#D97757" style="font-size:11px;font-weight:700;">AL dual ascent on &#955;</text>

  <circle cx="520" cy="302" r="3" fill="#7F1084"/>
  <text x="425" y="332" text-anchor="middle" fill="#7F1084" style="font-size:11px;font-weight:700;">back-propagation updates branch and trunk</text>
</svg>

</div>

<div class="text-[10px] px-1" style="color:#9CA3AF;">
Tensor shapes. Sensor path fixed per trajectory (201 time steps × 100 sensors); query path batched over
<b>N<sub>q</sub></b> = 1 024 collocation points per training step, 128² grid at evaluation.
</div>

</div>

<div class="mt-2" style="font-size:0.88rem; color:#374151;">
<b style="color:#7F1084;">PI-CON</b> — <b>P</b>hysics-<b>I</b>nformed <b>C</b>ontinuous-time <b>O</b>perator <b>N</b>etwork: a DeepONet readout with a continuous-time sensor branch, a distance-biased cross-attention fusion, and an Augmented-Lagrangian continuity constraint.
</div>

<FooterLogos />

<!--
[架構 · 1.5min]
• 深藍＝DeepONet 原有骨架／橘＝本研究加入（Fourier embed、CfC、cross-attn）
• 兩條路徑 + 紫線 training loop
• PI-CON 名稱首次登場
• Fourier embedding 並非原生 DeepONet；B0 保留它只是為了 matched-budget comparison
⚠️ 只講「是什麼」，為什麼在下一頁
-->

---

<NavBar active="method" />

<SectionTag>§ Methodology · why each addition is needed</SectionTag>

# What each addition is for

<style>
.ad { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 16px;
      align-items: stretch; }
.ad > div { display: flex; flex-direction: column; }
.ad .n { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: #D97757; }
.ad .t { font-size: 1.0rem; font-weight: 700; color: #1F1B2E; margin-top: 3px; }
.ad ul { margin-top: 8px; padding-left: 1.05em; }
.ad li { font-size: 0.87rem; color: #374151; line-height: 1.4; margin-bottom: 5px; list-style: disc; }
.ad li::marker { color: #C9C6D0; }
.ad li b { color: #1F1B2E; }
.ad .g { font-size: 0.78rem; color: #9CA3AF; margin-top: auto; padding-top: 6px; border-top: 1px solid #EFEAF2; }
</style>

<div class="ad">

<div>
<div class="n">Addition 1</div>
<div class="t">CfC branch</div>
<ul>
<li>Reads each sensor's <b>time history</b></li>
<li>Differentiable on an <b>uneven clock</b></li>
<li>Sensor record is an input, not a loss term</li>
</ul>
<div class="g">Closes the sensor-input and uneven-clock gaps</div>
</div>

<div>
<div class="n">Addition 2</div>
<div class="t">Cross-attention</div>
<ul>
<li>Maps <b>sparse sensors to any query point</b></li>
<li>Distance bias — near sensors weigh more</li>
<li>No fixed grid</li>
</ul>
<div class="g">Closes the sparse-to-dense gap</div>
</div>

<div>
<div class="n">Addition 3</div>
<div class="t">Augmented Lagrangian</div>
<ul>
<li><b>Adaptive penalty on ∇·u</b></li>
<li>Tightens as training proceeds</li>
<li>No hand-set weight</li>
</ul>
<div class="g">Keeps the field physical without a reference</div>
</div>

</div>

<div class="mt-6 px-4 py-3 rounded" style="background: rgba(127,16,132,0.06); border-left: 4px solid #7F1084; font-size:0.92rem; line-height:1.5; color:#374151;">
At K = 100 a vanilla DeepONet does reconstruct, but at <b>8.23 %</b> KE it misses the <b>10 %</b> engineering target. The three additions reach <b style="color:#7F1084;">5.71 %</b> <span style="color:#9CA3AF;">(p = 3 × 10⁻⁷)</span>.
</div>

<FooterLogos />

<!--
[三項新增各自解決什麼 · 1.5min]
• CfC → Gap 2/3・cross-attn → Gap 2・AL → Gap 1
⚠️ 不可說 vanilla DeepONet「做不出來」—— B0 = 8.23 % 可訓練，只是不達標
-->


---

<NavBar active="method" />

<SectionTag>§ Method · CfC branch (closing the time-signal gap)</SectionTag>

# CfC — closing the "time-signal" gap in vanilla DeepONet

<script setup>
const gateX = Array.from({ length: 31 }, (_, i) => i * 0.15)
const gateData = {
  labels: gateX.map(d => d.toFixed(1)),
  datasets: [{
    data: gateX.map(dt => 1 / (1 + Math.exp(dt - 2))),
    borderColor: '#7F1084', pointRadius: 0, fill: false,
  }],
}
const gateOpts = {
  scales: {
    x: { title: { display: true, text: 'Δt →', color: '#6B7280', font: { size: 8 } },
         ticks: { display: false }, grid: { display: false } },
    y: { title: { display: true, text: 'σ', color: '#7F1084', font: { size: 8 } },
         min: 0, max: 1, ticks: { stepSize: 1, color: '#6B7280', font: { size: 8 } },
         grid: { display: false } },
  },
  plugins: { legend: { display: false } },
}
</script>

<div class="grid grid-cols-2 gap-5 mt-2 text-sm">

<Card>
<LabelTiny>① LIQUID NEURAL NETWORK (LNN) [Hasani 2021]</LabelTiny>

<div class="mt-1 text-xs leading-snug">
h relaxes toward a target A, the <b>decay rate depends on the input</b> — a "liquid" time constant:
</div>

<div class="mt-1" style="font-size: 0.88em;">

$$\frac{d h}{dt} = -\underbrace{\Bigl[\tfrac{1}{\tau} + f(\cdot)\Bigr]}_{\text{input-dependent rate}} \odot\, h \;+\; f(\cdot) \odot A$$

</div>

<div class="mt-1 text-xs leading-snug" style="color:#6B7280;">
τ, A learnable, f(, ) a small MLP, <b style="color:#E97132;">✗ ODE solver in autograd is expensive</b>
</div>
</Card>

<Card>
<LabelTiny>② CfC — closed-form solution [Hasani 2022]</LabelTiny>

<div class="mt-1 text-xs leading-snug">
Same dynamics solved analytically — <b>a gate σ that blends two candidate states</b>:
</div>

<div class="mt-1" style="font-size: 0.88em;">

$$h(t + \Delta t) = \sigma \odot f_1 + (1 - \sigma) \odot f_2$$

</div>

<div class="mt-1" style="font-size: 0.88em;">

$$\sigma = \mathrm{sigmoid}(-\tau_a \Delta t + t_b)$$

</div>

<div class="mt-2 flex items-center gap-3">
<div style="width: 104px; flex: none;">
<ChartCanvas type="line" :data="gateData" :options="gateOpts" height="54px" />
</div>
<div class="text-xs leading-snug" style="color:#6B7280;">
short gap → <b style="color:#7F1084;">σ → 1</b> → f<sub>1</sub> (fast)<br/>
long gap → <b style="color:#7F1084;">σ → 0</b> → f<sub>2</sub> (relaxed)
</div>
</div>

<div class="mt-2 text-xs leading-snug" style="color:#374151;">
f<sub>1</sub> fast-response, f<sub>2</sub> slow-relaxation, <b style="color:#0F2D52;">✓ no ODE solver</b>, O(1)/step, autograd-safe
</div>
</Card>

</div>

<FooterLogos />

<!--
[CfC · backup 1min]
• branch 原本吃固定 grid，我們的 sensor 是不等間隔時序
• 卡1 ODE solver 在 autograd 內貴／卡2 閉式解 O(1)、autograd 安全
-->

---

<NavBar active="method" />

<SectionTag>§ Method · cross-attention readout (closing the sparse-to-dense gap)</SectionTag>

# Cross-attention — closing the "sparse-to-dense" gap

<div class="grid gap-5 mt-2 text-sm" style="grid-template-columns: 1.12fr 0.88fr;">

<Card>
<LabelTiny>① CROSS-ATTENTION READOUT [VASWANI 2017]</LabelTiny>

<div class="mt-1 text-xs leading-snug" style="color:#374151;">
For each target point <b style="color:#0F2D52;">q = (x, t)</b>, attention asks: <b>which sensors are most informative here?</b>
</div>

<div class="mt-2" style="font-size: 0.88em;">

$$A_{qk}=\operatorname{softmax}_k\!\left(\mathbf Q_q^{\mathsf T}\mathbf K_k/\sqrt{d}+b(r_{qk})\right)$$

</div>

<div class="mt-1 text-xs" style="display:grid; grid-template-columns:max-content 1fr; column-gap:9px; row-gap:2px; align-items:baseline; color:#374151;">
<b style="color:#0F2D52;">Q<sub>q</sub><sup>T</sup>K<sub>k</sub>/√d</b><span>learned query–sensor similarity</span>
<b style="color:#D97757;">b(r<sub>qk</sub>)</b><span>periodic spatial-distance bias</span>
</div>

<div class="mt-1" style="font-size: 0.88em;">

$$\mathbf c(q)=\sum_{k=1}^{K} A_{qk}\,\underbrace{\mathbf V_k}_{\text{sensor state at } t\,\le\, t_q}$$

</div>

<div class="mt-1 px-2 py-1 rounded text-xs" style="background:#FAF2FB; color:#374151; text-align:center;">
<b style="color:#7F1084;">Compare</b> q with all sensors <span style="color:#C9C6D0;">→</span>
<b style="color:#7F1084;">normalise</b> scores into A<sub>qk</sub> <span style="color:#C9C6D0;">→</span>
<b style="color:#7F1084;">combine</b> V<sub>k</sub> into c(q)
</div>

</Card>

<Card>
<LabelTiny>② TWO FLUID-SPECIFIC MODIFICATIONS</LabelTiny>

<div class="mt-1 text-xs leading-snug"><b>Causal lookup</b> <span style="color:#9CA3AF;">— picks K, V <i>before</i> attention</span> → <b style="color:#0F2D52;">streaming-deployable</b></div>

<svg viewBox="0 0 300 60" class="w-full mt-1">
  <rect x="6" y="16" width="181" height="21" fill="#7F1084" opacity="0.08" rx="2"/>
  <line x1="6" y1="37" x2="286" y2="37" stroke="#D1D5DB" stroke-width="1"/>
  <polygon points="286,33 295,37 286,41" fill="#D1D5DB"/>
  <text x="284" y="28" style="font-size:12px;font-style:italic" fill="#9CA3AF">t</text>
  <g fill="#7F1084">
    <circle cx="24" cy="37" r="3.4"/><circle cx="51" cy="37" r="3.4"/><circle cx="78" cy="37" r="3.4"/>
    <circle cx="105" cy="37" r="3.4"/><circle cx="132" cy="37" r="3.4"/><circle cx="159" cy="37" r="3.4"/>
  </g>
  <g fill="#fff" stroke="#D1D5DB" stroke-width="1.5">
    <circle cx="214" cy="37" r="3.4"/><circle cx="241" cy="37" r="3.4"/><circle cx="268" cy="37" r="3.4"/>
  </g>
  <line x1="187" y1="12" x2="187" y2="45" stroke="#D97757" stroke-width="2"/>
  <text x="191" y="21" style="font-size:12px;font-weight:700" fill="#D97757">query t<tspan style="font-size:9px" dy="2">q</tspan></text>
  <text x="6" y="55" style="font-size:12px;font-weight:700" fill="#7F1084">reads these</text>
  <text x="208" y="55" style="font-size:12px" fill="#9CA3AF">future — hidden</text>
</svg>

<div class="mt-1 text-xs leading-snug"><b>Isotropic bias</b> — distance decides, direction does not</div>

<svg viewBox="0 0 300 104" class="w-full mt-1">
  <g fill="none" stroke="#E5E7EB" stroke-width="1" stroke-dasharray="3 2">
    <circle cx="70" cy="46" r="18"/><circle cx="70" cy="46" r="34"/>
  </g>
  <g stroke="#C9B3D6" stroke-width="1">
    <line x1="70" y1="46" x2="98" y2="27"/><line x1="70" y1="46" x2="38" y2="58"/>
  </g>
  <text x="84" y="41" style="font-size:12px;font-style:italic" fill="#9CA3AF">r</text>
  <text x="50" y="61" style="font-size:12px;font-style:italic" fill="#9CA3AF">r</text>
  <circle cx="98" cy="27" r="5.5" fill="#7F1084"/>
  <circle cx="38" cy="58" r="5.5" fill="#7F1084"/>
  <circle cx="70" cy="46" r="3.5" fill="#D97757"/>
  <text x="70" y="99" style="font-size:12px;font-weight:700" fill="#0F2D52" text-anchor="middle">same r, same bias</text>
  <g transform="translate(230,46) rotate(-34)">
    <ellipse rx="40" ry="13" fill="#9CA3AF" opacity="0.12"/>
    <ellipse rx="40" ry="13" fill="none" stroke="#D1D5DB" stroke-width="1.2" stroke-dasharray="3 2"/>
  </g>
  <circle cx="258" cy="27" r="7.5" fill="#9CA3AF" opacity="0.9"/>
  <circle cx="198" cy="58" r="2.4" fill="#9CA3AF" opacity="0.55"/>
  <circle cx="230" cy="46" r="3.5" fill="#D97757"/>
  <text x="230" y="99" style="font-size:12px" fill="#9CA3AF" text-anchor="middle">direction decides</text>
</svg>

</Card>

</div>

<FooterLogos />

<!--
[Cross-attention · backup 1min]
• cross 不是 self：Q 來自 trunk，K/V 來自 sensor token
• 兩項修改：isotropic bias、causal lookup
⚠️ 因果性是**索引**給的不是 attention —— softmax 無 mask。問就答 searchsorted
⚠️ 不用 directional：會學成假方向性
-->


---

<NavBar active="method" />

<SectionTag>§ Training · closing the physics-consistency gap</SectionTag>

# Augmented Lagrangian on ∇·u

<style>
.al-proof { margin-top:9px; }
.al-proof .pair { display:grid; grid-template-columns:1fr max-content 1fr; gap:8px; align-items:center; margin-top:18px; }
.al-proof .case { padding:11px 7px; border:1px solid #E5E0EC; border-radius:8px; background:rgba(255,255,255,.65); text-align:center; }
.al-proof .case .tag { color:#6B7280; font-size:.63rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase; }
.al-proof .case .value { margin-top:4px; color:#374151; font-size:1.55rem; font-weight:700; line-height:1; font-variant-numeric:tabular-nums; }
.al-proof .case.ours { border-color:#C9A6CC; background:#FAF2FB; }
.al-proof .case.ours .tag, .al-proof .case.ours .value { color:#7F1084; }
.al-proof .equals { color:#7F1084; font-size:1.35rem; font-weight:700; }
.al-proof .hero-label { margin-top:7px; text-align:center; color:#6B7280; font-size:.73rem; }
.al-proof .check { margin-top:16px; padding:12px; border:1px solid #C9A6CC; border-radius:8px; background:#FAF2FB; }
.al-proof .check .label { color:#7F1084; font-size:.66rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
.al-proof .check .text { margin-top:4px; color:#374151; font-size:.86rem; font-weight:600; line-height:1.3; }
.al-proof .role { margin-top:12px; padding:9px 10px; border-left:3px solid #7F1084; background:#F7F4F8; color:#374151; font-size:.78rem; line-height:1.3; }
.al-proof .caveat { margin-top:9px; text-align:center; color:#9CA3AF; font-size:.68rem; line-height:1.25; }
</style>

<div class="grid grid-cols-2 gap-5 mt-3 text-sm">

<Card>
<LabelTiny>AUGMENTED LAGRANGIAN (AL) ON CONTINUITY</LabelTiny>

<div class="mt-2" style="font-size: 0.94em;">

$$\mathcal{L}_{\text{AL}} \;=\; \lambda\,\mathcal{L}_{\text{cont}} \;+\; \tfrac{0.1}{2}\,\mathcal{L}_{\text{cont}}^{\,2}$$

</div>

<div class="mt-1" style="font-size: 0.94em;">

$$\mathcal{L}_{\text{cont}} \,=\, \mathbb{E}_{\text{collocation}}\big[(\partial_x u + \partial_y v)^2\big] \,\ge\, 0$$

</div>

<div class="mt-1" style="font-size: 0.94em;">

$$\lambda \,\leftarrow\, \mathrm{clip}\big(\lambda + 0.1\,\overline{\mathcal{L}}_{\text{cont}},\; 0,\; 10\big)$$

</div>

<div class="mt-3 pt-2" style="border-top:1px solid #E5E0EC; color:#6B7280; font-size:.74rem; line-height:1.3;">
<span class="raw">𝓛<sub>cont</sub></span>: continuity residual &nbsp;·&nbsp; <span class="raw">λ</span>: dual multiplier &nbsp;·&nbsp; <span class="raw" style="text-decoration:overline;">𝓛</span><span class="raw"><sub>cont</sub></span>: EMA (<span class="raw">β</span> = 0.5)
</div>

<div class="mt-2" style="color:#E97132; font-size:.82rem; font-weight:700; line-height:1.25;">
<span class="raw">𝓛<sub>cont</sub></span> ≥ 0 &nbsp;⇒&nbsp; <span class="raw">λ</span> only increases.
</div>
</Card>

<Card>
<LabelTiny>CONTINUITY DIAGNOSTIC</LabelTiny>

<div class="al-proof">
<div class="pair">
<div class="case">
<div class="tag">DNS reference</div>
<div class="value">0.38 %</div>
</div>
<div class="equals">≈</div>
<div class="case ours">
<div class="tag">PI-CON</div>
<div class="value">0.39 %</div>
</div>
</div>
<div class="hero-label">Divergence ratio · PI-CON: five training seeds</div>

<div class="check">
<div class="label">What this checks</div>
<div class="text">PI-CON stays at the matched DNS reference level, confirming that the AL continuity constraint is active.</div>
</div>

<div class="role"><b style="color:#7F1084;">Role in this study:</b> a physical-consistency diagnostic alongside reconstruction accuracy.</div>
<div class="caveat">A diagnostic comparison—not a full-field fidelity metric.</div>
</div>

</Card>

</div>

<FooterLogos />

<!--
[AL on continuity · 1.5min]
• 右卡只要讀 0.38% ≈ 0.39%：PI-CON 的 divergence ratio 對上 matched DNS reference
• 因此可說 AL continuity constraint active；這是 physical-consistency diagnostic，不是全場精度指標
⚠️ 問 SIMPLE/PISO（版上已無，只能口述）：
　 ① 它逐點投影，我們是平均意義　② 逐點不可微、進不了 GradNorm
⚠️ 絕不可說 sub-DNS 或「比 DNS 更不可壓縮」
⚠️ Λ_max 從未撞到（λ 停在 0.386 = clip 的 3.9 %）
• 問 ρ → 拉到 1 可把 div 壓到 0.28 %，但犧牲場精度
-->

---

<NavBar active="method" />

<SectionTag>§ Training · convergence &amp; loss balance</SectionTag>

# Optimization: second-order updates and loss balance

<style>
.opt-frame { margin-top:16px; }
.opt-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.opt-grid .hero { color:#7F1084; font-size:1.62rem; font-weight:700; line-height:1.05; }
.opt-grid .subtitle { margin-top:3px; color:#6B7280; font-size:.76rem; font-weight:600; letter-spacing:.04em; text-transform:uppercase; }
.opt-grid .ref { color:#9CA3AF; font-weight:400; letter-spacing:0; text-transform:none; white-space:nowrap; }
.opt-grid .mechanism { display:grid; grid-template-columns:1fr max-content 1fr max-content 1fr; gap:7px; align-items:center; margin-top:18px; }
.opt-grid .mechanism .node { min-height:72px; display:flex; align-items:center; justify-content:center; padding:7px; border:1px solid #E5E0EC; border-radius:8px; background:#fff; color:#374151; font-size:.73rem; line-height:1.22; text-align:center; }
.opt-grid .mechanism .node.emph { border-color:#C9A6CC; background:#FAF2FB; color:#7F1084; font-weight:700; }
.opt-grid .mechanism .arrow { color:#7F1084; font-size:1.05rem; font-weight:700; }
.opt-grid .takeaway { margin-top:15px; padding-top:10px; border-top:1px solid #E5E0EC; color:#374151; font-size:.82rem; line-height:1.32; }
.opt-grid .secondary { margin-top:12px; color:#9CA3AF; font-size:.70rem; line-height:1.25; }
.opt-grid .chips { display:flex; flex-wrap:wrap; justify-content:center; gap:6px; margin-top:13px; }
.opt-grid .chip { padding:3px 7px; border:1px solid #E5E0EC; border-radius:999px; color:#374151; font-size:.72rem; background:#fff; }
.opt-grid .equation { margin-top:14px; text-align:center; font-size:.90em; }
.opt-grid .equation :deep(.katex-display) { margin:.2em 0; }
</style>

<div class="opt-frame">
<div class="opt-grid">

<Card>
<LabelTiny>1 · CONVERGENCE</LabelTiny>
<div class="hero">SOAP</div>
<div class="subtitle">Shampoo-style 2nd-order preconditioner <span class="ref">[Vyas et al., 2024]</span></div>

<div class="mechanism">
<div class="node">gradient<br>statistics</div><div class="arrow">→</div>
<div class="node emph">Shampoo<br>preconditioner</div><div class="arrow">→</div>
<div class="node">Adam update<br>in its eigenbasis</div>
</div>

<div class="takeaway"><b style="color:#7F1084;">Role:</b> condition the anisotropic loss geometry before each update.</div>
<div class="secondary">Schedule-Free: auxiliary Polyak–Ruppert averaging.</div>
</Card>

<Card>
<LabelTiny>2 · LOSS BALANCE</LabelTiny>
<div class="hero">GradNorm</div>
<div class="subtitle">Adaptive weights for four loss tasks <span class="ref">[Chen et al., 2018]</span></div>

<div class="equation">

$$\|w_i\,\nabla\!_{\theta_r}\,\mathcal{L}_i\| \;\propto\; (\mathcal{L}_i / \mathcal{L}_i^{(0)})^{\alpha}$$

</div>

<div class="mechanism">
<div class="node">data<br>NS-u<br>NS-v<br>continuity</div><div class="arrow">→</div>
<div class="node emph">adapt<br><span class="raw">w<sub>i</sub></span></div><div class="arrow">→</div>
<div class="node">matched<br>gradient norms</div>
</div>
<div class="takeaway"><b style="color:#7F1084;">Every 1 000 steps:</b> rebalance the four training tasks.</div>
</Card>

</div>
</div>

<FooterLogos />

<!--
[最佳化 · 2min]
• 這頁只講兩件事：SOAP 管收斂，GradNorm 管多任務平衡
• SOAP 是 Shampoo-style 二階預條件器，處理 Re=10⁴ 下 anisotropic loss geometry；Schedule-Free 僅為次要 averaging wrapper
• GradNorm 每 1000 步重新平衡 data、NS-u、NS-v、continuity 四個任務的 gradient magnitude
⚠️ 不要報「−20 % KE」（thesis 未收）
-->

---

<NavBar active="method" />

<SectionTag>§ Model and training configuration</SectionTag>

# Model and training configuration

<style>
.pgrid { display: grid; grid-template-columns: max-content 1fr; column-gap: 20px; row-gap: 7px;
         font-size: 0.9rem; line-height: 1.35; margin-top: 10px; }
.pgrid .k { color: #7F1084; font-weight: 600; white-space: nowrap; }
.pgrid .v { color: #1F1B2E; }
.pgrid .cite { color: #9CA3AF; }
.dimtable { width:100%; border-collapse:collapse; table-layout:fixed; margin-top:10px; font-size:.72rem; line-height:1.2; }
.dimtable th { padding:0 6px 5px; color:#9CA3AF; font-size:.60rem; text-transform:uppercase;
               letter-spacing:.05em; text-align:left; border-bottom:1px solid #E5E0EC; }
.dimtable td { padding:7px 6px; color:#374151; border-bottom:1px solid #F1EDF5; vertical-align:top; }
.dimtable td:first-child { color:#7F1084; font-weight:700; }
.dimtable td:last-child { color:#6B7280; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
                          font-size:.66rem; white-space:nowrap; }
.dimsum { display:flex; justify-content:space-between; gap:12px; margin-top:8px; padding-top:7px;
          border-top:1px solid #E5E0EC; color:#374151; font-size:.72rem; }
.dimsum b { color:#7F1084; }
</style>

<div class="grid gap-6 mt-3" style="grid-template-columns:1.1fr 0.9fr;">

<Card>
<LabelTiny>Network size</LabelTiny>
<table class="dimtable">
<colgroup><col style="width:34%"><col style="width:30%"><col style="width:36%"></colgroup>
<thead><tr><th>Module</th><th>Size</th><th>Tensor</th></tr></thead>
<tbody>
<tr><td>Sensor encoder</td><td>1 residual block</td><td>201 × 100 × 256</td></tr>
<tr><td>CfC branch</td><td>1 layer + 2 attention<br>blocks, 4 heads</td><td>201 × 100 × 256</td></tr>
<tr><td>Fourier embed</td><td>16 harmonics</td><td>queries × 128</td></tr>
<tr><td>MLP trunk</td><td>1 residual block</td><td>queries × 256</td></tr>
<tr><td>Cross-attention</td><td>1 layer, 1 head</td><td>queries × 100 × 256</td></tr>
<tr><td>Inner product</td><td>3 fields × rank 256</td><td>queries × 3</td></tr>
</tbody>
</table>
<div style="margin-top:7px; font-size:.64rem; line-height:1.35; color:#9CA3AF;">
201 = snapshots <span style="color:#D8D2E0;">·</span> 100 = sensors <span style="color:#D8D2E0;">·</span>
256 = feature width <span style="color:#D8D2E0;">·</span> 3 = (u, v, p)
</div>

<div class="dimsum">
<span>Total trainable parameters <b>3.14 M</b></span>
</div>
</Card>

<Card>
<LabelTiny>Training parameters</LabelTiny>
<div class="pgrid">
<div class="k">Collocation</div><div class="v">1 024 points / optimizer step</div>
<div class="k">Iterations</div><div class="v">20 000 / run</div>
<div class="k">LR warm-up</div><div class="v">2 000 steps</div>
<div class="k">Time marching</div><div class="v">2 000 steps &nbsp;(<span class="raw">t</span>: 0.5 → 5)</div>
<div class="k">Seeds</div><div class="v">42, 1, 2, 3, 4 &nbsp;(<b>n = 5</b> independent runs)</div>
<div class="k">Reported</div><div class="v">mean ± 1σ over n = 5 runs</div>
<div class="k">Hardware</div><div class="v"><b style="color:#7F1084;">Single</b> RTX 3090 (24 GB), ~2 h 45 m / run</div>
</div>
</Card>

</div>


<FooterLogos />

<!--
[模型與訓練配置 · 1min]
• 左邊交代模型尺寸；右邊只保留實驗預算與計算條件
• 每一列對應架構圖的一個方塊
⚠️ 2 個 attention block 屬 CfC branch，不是 spatial encoder
• 備答：inner 512・SOAP β=(0.9,0.999)・GradNorm init (1,.01,.01,.01)・AL ρ=0.1 clip=10
-->

---

<NavBar active="method" />

<SectionTag>§ Evaluation metrics &amp; training loss</SectionTag>

# Error metrics & training loss

<style>
.ngrid { display: grid; grid-template-columns: max-content 1fr; column-gap: 16px; row-gap: 4px; align-items: baseline; margin-top: 6px; }
.ngrid .sym { color: #7F1084; font-weight: 600; font-size: 0.82rem; white-space: nowrap; }
.ngrid .def { color: #374151; font-size: 0.82rem; line-height: 1.25; }
.eqbox { border-left: 2px solid #E5E0EC; padding-left: 12px; margin: 4px 0 2px 0; font-size: 0.72em; }
</style>

<div class="grid grid-cols-2 gap-5 mt-3">

<Card>
<LabelTiny>FIELD ERROR METRICS &nbsp;<span class="opacity-60">(offline DNS benchmark only)</span></LabelTiny>

<div class="eqbox">

$$\mathrm{rel}\,L_2(\phi) = \frac{\|\phi_{\text{pred}} - \phi_{\text{DNS}}\|_2}{\|\phi_{\text{DNS}}\|_2}, \quad \phi \in \{u, v, \omega\}$$

</div>

<div class="eqbox" style="border-left-color:#7F1084;">

$$\overline{\mathrm{KE\_MAPE}} = \frac{1}{|T|}\sum_{t}\frac{\bigl|\mathrm{KE}_{\text{pred}}(t) - \mathrm{KE}_{\text{DNS}}(t)\bigr|}{\mathrm{KE}_{\text{DNS}}(t)}$$

</div>

<div class="ngrid">
<div class="sym">rel-L₂(u,v,ω)</div><div class="def">global field error</div>
<div class="sym">KE MAPE</div><div class="def"><b>headline</b>; mean absolute percentage error of KE(t) = ½∫<sub>Ω</sub>(u²+v²) dx</div>
<div class="sym">RMSE<sub>u,v</sub>(t)</div><div class="def">time-resolved, absolute (not normalised)</div>
<div class="sym">div ratio</div><div class="def">‖∇·u‖₂ / ‖∇u‖<sub>F</sub><sup>DNS</sup></div>
</div>

<div class="foot mt-1">Derivatives: 2nd-order central differences, 128² grid.</div>
</Card>

<Card>
<LabelTiny>TRAINING LOSS &nbsp;<span class="opacity-60">(GradNorm-balanced [Chen 2018])</span></LabelTiny>

<div class="eqbox">

$$\mathcal{L}(\theta) = w_d \mathcal{L}_{\text{data}} + w_{\text{NS},u} \mathcal{L}_{\text{NS},u} + w_{\text{NS},v} \mathcal{L}_{\text{NS},v} + w_c \mathcal{L}_{\text{cont}} + \textcolor{#E97132}{\mathcal{L}_{\text{AL}}}$$

</div>

<div class="eqbox" style="border-left-color:#E97132;">

$$\mathcal{L}_{\text{AL}} = \lambda\,\mathcal{L}_{\text{cont}} + \tfrac{\rho}{2}\,\mathcal{L}_{\text{cont}}^{\,2}$$

</div>

<div class="ngrid">
<div class="sym">ℒ<sub>data</sub></div><div class="def">MSE on the K = 100 sensor channels</div>
<div class="sym">ℒ<sub>NS,u</sub> , ℒ<sub>NS,v</sub></div><div class="def">NS momentum residual at collocation points</div>
<div class="sym">ℒ<sub>cont</sub></div><div class="def">∇·u residual — the AL acts on this same term</div>
<div class="sym" style="color:#E97132;">ℒ<sub>AL</sub></div><div class="def">adaptive continuity pressure via λ, <b>outside</b> GradNorm</div>
<div class="sym">w<sub>d</sub> , w<sub>NS</sub> , w<sub>c</sub></div><div class="def">GradNorm-balanced weights</div>
</div>

<div class="mt-3 pt-2 text-xs leading-snug" style="border-top: 1px solid #E5E0EC; color:#374151;">
<b style="color:#7F1084;">No full field enters ℒ</b> — not the DNS, not the LES.
</div>
</Card>

</div>

<FooterLogos />

<!--
[誤差指標與 loss · 1.5min]
• 四個量：global rel-L₂／KE MAPE／逐時 RMSE／div ratio
• 底線：**DNS 或 LES 全場都不進 L**。LES 只決定 sensor 放哪裡，不是訓練資料
⚠️ continuity 進 loss **兩次**（GradNorm + AL）；AL 刻意放在 GradNorm 之外
-->

---

<NavBar active="results" />

<SectionTag>§ Results · main result · architectural value</SectionTag>

# Main result — 2×2 ablation over 5 random seeds

<div class="text-[10px] mt-1" style="color:#6B7280;">
Re = 10⁴, K = 100, LES-derived QR-pivot placement (DNS-free), 1024 collocation, 20 k iterations, all cells n = 5 seeds
</div>

<style>
.abmatrix { display: grid; grid-template-columns: max-content minmax(0, 1fr) minmax(0, 1fr) max-content; column-gap: 10px; row-gap: 5px; align-items: center; margin-top: 6px; margin-bottom: 0; }
.abmatrix .hd { font-size: 0.78rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.03em; text-align: center; }
.abmatrix .rl { font-size: 0.82rem; color: #6B7280; white-space: nowrap; }
.abmatrix .mg { font-size: 0.72rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.03em; white-space: nowrap; text-align: center; }
.abmatrix .cell { border: 1px solid #E5E0EC; border-radius: 6px; padding: 7px 4px; text-align: center; background: #FFF; }
.abmatrix .cell.best { border-color: #7F1084; background: #FAF3FB; }
.abmatrix .id { display: block; font-size: 0.90rem; color: #9CA3AF; letter-spacing: 0.05em; white-space: nowrap; }
.abmatrix .val { display: block; font-size: 1.05rem; font-weight: 700; color:#1F1B2E; line-height: 1.15; }
.abmatrix .cell.best .val { color: #7F1084; }
.abmatrix .dv { font-size: 0.90rem; font-weight: 700; text-align: center; }
.abmatrix .good { color: #7F1084; }
.abmatrix .bad  { color: #E97132; }
.rg { display: grid; grid-template-columns: 1fr max-content; column-gap: 12px; row-gap: 4px; align-items: baseline; margin-top: 6px; margin-bottom: 0; }
.rg .k { font-size: 0.90rem; color: #374151; }
.rg .n { font-size: 0.90rem; font-weight: 700; text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; color: #1F1B2E; }
.rg .tot { border-top: 1px solid #E5E0EC; padding-top: 5px; margin-top: 2px; }
</style>

<div class="grid gap-3 mt-1 items-start" style="grid-template-columns: minmax(0, 1.72fr) minmax(0, 1.28fr);">

<div>
<Card>
<LabelTiny>KE MAPE (%) &nbsp;<span class="opacity-60">mean over 5 seeds, lower is better</span></LabelTiny>

<div class="abmatrix">
<div></div>
<div class="hd">no cross-attn</div>
<div class="hd">+ cross-attn</div>
<div class="mg">Δ from<br/>cross-attn</div>

<div class="rl">no CfC</div>
<div class="cell"><span class="id">B0</span><span class="val">8.23</span></div>
<div class="cell"><span class="id">B2</span><span class="val">7.03</span></div>
<div class="dv good">−1.21</div>

<div class="rl">+ CfC</div>
<div class="cell"><span class="id">B1</span><span class="val">9.23</span></div>
<div class="cell best"><span class="id">B3 &nbsp;PI-CON</span><span class="val">5.71</span></div>
<div class="dv good">−3.52</div>

<div class="mg">Δ from CfC</div>
<div class="dv bad">+0.99</div>
<div class="dv good">−1.32</div>
<div></div>
</div>

<div class="foot mt-2">Bottom row flips sign: CfC costs <b style="color:#E97132;">+0.99</b> alone, buys <b style="color:#7F1084;">−1.32</b> with cross-attention. Contrasts use unrounded seed means.</div>
</Card>
</div>

<div class="space-y-2 text-xs">

<Card>
<LabelTiny>KE decomposition &nbsp;<span class="opacity-60">(pp)</span></LabelTiny>
<div class="rg">
<div class="k">cross-attention</div><div class="n" style="color:#7F1084;">−1.21</div>
<div class="k">CfC</div><div class="n" style="color:#E97132;">+0.99</div>
<div class="k">CfC × cross-attention</div><div class="n" style="color:#7F1084;">−2.31</div>
<div class="k tot">total &nbsp;B3 − B0</div><div class="n tot">−2.53</div>
</div>
</Card>

<Card>
<LabelTiny>Welch <span class="raw">t</span>-test &nbsp;<span class="opacity-60">5 random seeds per cell</span></LabelTiny>
<div class="rg">
<div class="k">B3 &nbsp;PI-CON</div><div class="n" style="color:#7F1084;">5.71 ± 0.12 %</div>
<div class="k">B0 &nbsp;vanilla DeepONet</div><div class="n">8.23 ± 0.22 %</div>
<div class="k tot">gap</div><div class="n tot" style="color:#7F1084;">−30.7 % rel</div>
<div class="k">significance</div><div class="n" style="color:#7F1084;">p = 3 × 10⁻⁷</div>
</div>
</Card>

</div>

</div>

<FooterLogos />

<!--
[架構 ablation · 2min]
• 分解（口述）：cross-attn −1.21・CfC +0.99・interaction −2.31・合計 −2.53 pp
• 統計：t = 22.9、p = 3×10⁻⁷、d = 14.5
• ① 兩個元件都必要　② operator 勝過純容量（PINN 3.24 M vs 1.28 M）
-->

---

<NavBar active="results" />

<div class="grid grid-cols-5 gap-4 mt-2">

<div class="col-span-2">

<SectionTag>§ Results · field reconstruction at t = 5</SectionTag>

# Field reconstruction<br/><span style="font-size: 0.85em; color:#6B7280;">ω, u, v at t = 5</span>

<Card>
<LabelTiny>KEY OBSERVATIONS</LabelTiny>
<div class="mt-2 text-xs leading-snug space-y-1">
<div>• Main vortex structure recovered</div>
<div>• Small scales (k &gt; 5) smoothed — sensor Nyquist scale</div>
<div>• Error sits on <b>high-shear edges</b>, not random</div>
<div>• |u, v error| ≪ |ω error| (ω amplifies derivatives)</div>
</div>
</Card>

<div class="mt-3 text-xs leading-snug" style="color:#6B7280;">
Source: PI-CON baseline (B3 + LES_T50 + 1024 collocation points), seed 42 field visualization; metrics use n = 5.
</div>

</div>

<div class="col-span-3">
<Card style="padding-top: 0.5rem; padding-bottom: 0.5rem;">

<img :src="'/images/field_comparison_t5.png'" style="width: 100%; object-fit: contain;" />

<img :src="'/images/vorticity_comparison_t5.png'" class="mt-1" style="width: 100%; object-fit: contain;" />

</Card>
</div>

</div>

<FooterLogos />

<!--
[場重建 · 2.5min]
• velocity 幾乎一致；vorticity 是導數量，放大誤差但主結構仍在
• 誤差在 high-shear edges，不是隨機雜訊
⚠️ 顏色：DNS 與 PI-CON 共用 ±max，error panel 獨立縮放
-->

---

<NavBar active="results" />

<SectionTag>§ Results · vorticity error interpretation</SectionTag>

# Error structure across wavenumbers

<style>
.bg2 { display: grid; grid-template-columns: max-content 1fr; column-gap: 14px; row-gap: 4px; align-items: baseline; margin-top: 6px; margin-bottom: 0; }
.bg2 .k { font-size: 0.90rem; color: #1F1B2E; white-space: nowrap; }
.bg2 .v { font-size: 0.90rem; color: #1F1B2E; line-height: 1.3; }
.bg2 .omega { color:#7F1084; font-weight:700; }
</style>

<div class="grid grid-cols-5 gap-4 mt-3">

<div class="col-span-3">
<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<LabelTiny>Band-resolved relative error vs time &nbsp;<span class="opacity-60">(main baseline, n = 5)</span></LabelTiny>
<img :src="'/images/band_energy_rel_error_vs_time.png'" class="mt-1" style="width: 100%; max-height: 248px; object-fit: contain;" />
</Card>
</div>

<div class="col-span-2 space-y-2">

<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<LabelTiny>Key metrics &nbsp;<span class="opacity-60">(main baseline, n = 5)</span></LabelTiny>
<div class="bg2">
<div class="k">KE MAPE</div><div class="v">5.71 ± 0.12 %</div>
<div class="k">u rel-L₂</div><div class="v">13.65 ± 0.06 %</div>
<div class="k">v rel-L₂</div><div class="v">17.52 ± 0.10 %</div>
<div class="k omega">ω rel-L₂</div><div class="v omega">41.79 ± 0.12 %</div>
<div class="k">div ratio</div><div class="v">0.39 ± 0.006 %</div>
</div>
</Card>

<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<LabelTiny>Why KE 5.7 % but <span class="raw">ω</span> 41.8 %</LabelTiny>
<div class="bg2">
<div class="k">k ≤ 5</div><div class="v">99 % of energy, err <b>2.5 %</b></div>
<div class="k">k, 5–16</div><div class="v"><b>53 %</b>, about half recovered</div>
<div class="k">k &gt; 16</div><div class="v"><b>99.9 %</b>, no energy placed</div>
</div>
<div class="mt-1 text-xs leading-snug" style="color:#6B7280;">KE weights energy; ω is broadband pointwise.</div>
</Card>

</div>

</div>

<FooterLogos />

<!--
[誤差的波數結構 · 2min]
• k ≤ 5 就是 k_max 5.64；越過後 κ 由 7 → 7×10²，加大網路補不回來
⚠️ 別說成「不可觀測」—— 那要到 k ≈ 8 才成立
-->

---

<NavBar active="results" />

<SectionTag>§ Results · main baseline (B3 + LES_T50, 1024 collocation points)</SectionTag>

# Temporal diagnostics

<div class="grid grid-cols-2 gap-4 mt-3">

<Card>
<img :src="'/images/kinetic_energy_vs_time.png'" class="rounded" style="max-height: 252px; width: 100%; object-fit: contain;" />
<div class="foot mt-1">KE(t) integrated over the <b>full domain</b>. MAPE <b style="color:#7F1084;">5.71 ± 0.12 %</b> (n = 5), follows DNS decay 0.161 → 0.122, IC warm-up t &lt; 2.</div>
</Card>

<Card>
<img :src="'/images/uv_rmse_vs_time.png'" class="rounded" style="max-height: 252px; width: 100%; object-fit: contain;" />
<div class="foot mt-1">u, v RMSE <b style="color:#7F1084;">0.115 → 0.03</b> (n = 5, ±1σ), absolute, no denominator, flat after t ≈ 3 s.</div>
</Card>

</div>

<FooterLogos />

<!--
[時序診斷 · 1.5min]
• KE 追 DNS 0.161 → 0.122；右圖是**絕對 RMSE**
⚠️ 為何不用 rel-L₂：DNS 分量重分配、‖v‖ 掉 37 % → 上翹是分母假象，不是重建變差
⚠️ 與主表 13.65/17.52 不同量（global vs 逐時）
⚠️ 重分配成因未確立（r = −0.26）
-->

---

<NavBar active="results" />

<SectionTag>§ Results · sensor placement axis (O3)</SectionTag>

# Sensor placement without DNS access

<style>
.plc { width:100%; border-collapse:collapse; margin-top:22px; font-size:1.05rem;
       font-variant-numeric:tabular-nums; }
.plc th { text-align:right; font-weight:700; color:#374151; font-size:0.92rem;
          padding:0 16px 12px; border-bottom:1.5px solid #1F1B2E; white-space:nowrap; }
.plc th:first-child { text-align:left; }
.plc td { padding:16px 16px; border-bottom:1px solid #E5E0EC; color:#374151; text-align:right; }
.plc td:first-child { text-align:left; color:#1F1B2E; }
.plc tr:last-child td { border-bottom:1.5px solid #1F1B2E; }
.plc .sub { font-weight:400; color:#9CA3AF; font-size:0.88em; }
.plc tr.main td:first-child { font-weight:700; }
</style>

<table class="plc">
<thead>
<tr>
<th>Placement strategy</th>
<th>DNS field required</th>
<th>KE MAPE (%)</th>
<th><span class="raw">u</span> rel-L₂ (%)</th>
<th><span class="raw">v</span> rel-L₂ (%)</th>
</tr>
</thead>
<tbody>
<tr class="main">
<td>LES <span class="raw">T</span> = 50 <span class="sub">main pipeline</span></td>
<td>No</td>
<td>5.71 ± 0.12</td>
<td>13.65</td>
<td>17.52</td>
</tr>
<tr>
<td>DNS QR-pivot <span class="sub">oracle</span></td>
<td>Yes</td>
<td>4.68 ± 0.06</td>
<td>15.34</td>
<td>18.10</td>
</tr>
<tr>
<td>Random uniform <span class="sub">fallback</span></td>
<td>No</td>
<td>7.95 ± 0.76</td>
<td>17.20</td>
<td>21.62</td>
</tr>
</tbody>
</table>

<div class="foot mt-3">Same B3 backbone, K = 100, 1024 collocation points, 20 000 iterations. LES/DNS rows vary <b>training seed</b> at fixed placement; Random varies <b>placement seed</b> at fixed training seed.</div>
<div class="mt-2 text-sm font-semibold" style="color:#7F1084;">Placement variability is 6.4× training variability: 0.76 % vs 0.12 %.</div>

<FooterLogos />

<!--
[佈點 · 2min]
• 三種佈點的 KE MAPE 皆 < 10 % → feasibility；數值隨位置變 → reliability（O3）
• trade-off：oracle 贏 KE（4.68），但 LES 贏 pointwise（u/v 13.65/17.52 vs 15.34/18.10）
• LES vs random 差 −2.24 pp；placement variance ≈ 6.4× training variance
⚠️ 不可拿 LES 能譜比 DNS 當品質證據
-->

---

<NavBar active="results" />

<SectionTag>§ Results · sensor-only baselines and DNS oracle</SectionTag>

# Kinetic energy error versus pointwise accuracy

<style>
.fb { width: 100%; border-collapse: collapse; font-size: 0.98rem; margin-top: 8px; }
.fb th { text-align: right; font-weight: 700; color: #6B7280; font-size: 0.86rem; text-transform: uppercase;
         letter-spacing: 0.04em; padding: 0 10px 8px 10px; border-bottom: 1px solid #D8D2E0; }
.fb th.m { text-align: left; }
.fb td { padding: 7px 10px; border-bottom: 1px solid #F1EDF5; color: #374151; text-align: right;
         font-variant-numeric: tabular-nums; }
.fb td.m { text-align: left; color: #1F1B2E; white-space: nowrap; }
.fb tr.ours td { background: #F7EDF8; border-bottom: none; font-weight: 700; }
.fb tr.oracle td { background: #FFF7F1; border-top: 1px solid #F4C3A9; color:#374151; }
.fb tr.oracle td.m { color:#A84B18; font-weight:700; }
.fb .win { color: #7F1084; font-weight: 700; }
.fb .trap { color: #E97132; font-weight: 700; }
.setup { display:grid; grid-template-columns:max-content 1fr; column-gap:14px; row-gap:4px;
         margin-top:9px; font-size:.76rem; line-height:1.25; color:#374151; }
.setup .n { color:#7F1084; font-weight:700; white-space:nowrap; }
</style>

<table class="fb">
  <thead>
    <tr>
      <th class="m">Method &nbsp;<span style="font-weight:400; text-transform:none; letter-spacing:0;">(K = 100 sensors; oracle marked)</span></th>
      <th>KE %<br/><span style="font-weight:400; text-transform:none; letter-spacing:0; color:#E97132;">lower ≠ better</span></th><th><span class="raw">rel-L₂ of u</span></th><th><span class="raw">rel-L₂ of v</span></th><th><span class="raw">rel-L₂ of ω</span></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="m">Radial basis function (RBF)</td>
      <td class="trap">5.08</td><td>30.02</td><td>34.43</td><td>58.33</td>
    </tr>
    <tr>
      <td class="m">Inverse distance weighting (IDW)</td>
      <td>66.66</td><td>52.88</td><td>62.02</td><td>81.89</td>
    </tr>
    <tr>
      <td class="m">Divergence-free trigonometric least squares (trig-LSQ)</td>
      <td class="trap">4.42</td><td>25.87</td><td>31.96</td><td>63.41</td>
    </tr>
    <tr class="oracle">
      <td class="m">Gappy-POD, r = 100 <span style="font-weight:400; color:#E97132;">(val-only DNS time-interpolation oracle)</span></td>
      <td class="win">0.229</td><td class="win">1.70</td><td class="win">2.90</td><td>—</td>
    </tr>
    <tr class="ours">
      <td class="m">PI-CON <span style="color:#9CA3AF; font-weight:400;">(ours, n = 5)</span></td>
      <td>5.71</td><td class="win">13.65</td><td class="win">17.52</td><td class="win">41.77</td>
    </tr>
  </tbody>
</table>

<div class="setup">
<div class="n">RBF</div><div>multiquadric interpolation (<span class="raw">ε</span> = 10)</div>
<div class="n">IDW</div><div>inverse-distance weighting (<span class="raw">∝</span> 1/d²)</div>
<div class="n">trig-LSQ</div><div>divergence-free Fourier LSQ (k<sub>max</sub> = 5)</div>
<div class="n" style="color:#E97132;">Gappy-POD</div><div>DNS-trained POD oracle (r = 100; 160 full fields)</div>
</div>


<div class="foot mt-1">Validation: 41 held-out times; reconstruct from each time’s K sensors, not a forecast.</div>

<FooterLogos />

<!--
[經典內插對照 · 1.5min]
• 前三列與 PI-CON 才是「同 sensor・同指標」的工程比較
• r=100 是保留 100 個 joint (u,v) POD spatial modes，不是 100 個時間點
• Gappy-POD 每個 t 也只讀 K sensor；但 basis 已由 t∈[0,5] 的 160 個 interleaved DNS full fields 學好 → temporal-interpolation oracle，不是公平 head-to-head
• 41 個 val times 是每 0.125 一個；它用該時刻 sensor 重建，而非 t=0 free-run 到 t=5
• RBF 5.08／trig-LSQ 4.42 的 KE 更低，但 u rel-L₂ 是我們兩倍 → 收縮到 inter-sensor mean
• 口頭補：u rel-L₂ 降 47 %（vs trig-LSQ）／74 %（vs IDW）
⚠️ **47/74 只有 u 一個場**；三場平均是 42 % / 65 %
⚠️ 問「挑弱對手」→ 參數由 a-priori 定，不穩定變體已被排除，我們選的是強項組態
⚠️ 通用方法非某篇論文，不可說「贏了某某論文」
-->

---

<NavBar active="results" />

<SectionTag>§ Results · sensor count axis (O2)</SectionTag>

# K-scaling — cutoff vs. sensor count

<div class="text-sm mt-1" style="color:#374151;">
PI-CON departs from DNS at the sensor Nyquist <span class="raw">k<sub>max</sub> = √(K/π)</span> — fidelity comes from <b style="color:#7F1084;">bandwidth expansion</b>, not architecture search.
</div>

<Card style="padding: 0.45rem 0.6rem;" class="mt-2">
<img :src="'/images/spectrum_k_scaling_triptych.png'"
     style="display: block; margin: 0 auto; max-width: 87%; height: auto;" />
</Card>

<div class="mt-2 text-xs leading-snug" style="color:#6B7280;">
Single-seed, read as a trend, not a fit. The same budget argument carries across Reynolds number: <b style="color:#7F1084;">Re = 10⁶ with K = 200 reaches 6.10 %</b> <span style="color:#9CA3AF;">(feasibility check, single seed)</span>.
</div>

<FooterLogos />

<!--
[K-scaling · 1.5min]
• 指綠線 5.64 → 7.98 → 11.28 右移，PI-CON 正好在綠線處脫離 DNS
• 這也是 spectral-bias 的反駁
• Re=10⁶ 講 **feasibility 不是 benchmark**，也不說 generalisation
⚠️ single-seed trend，不是 fit
⚠️ 不要改畫長條圖（三點不共線）
-->


---

<NavBar active="results" />

<SectionTag>§ Results · sensor count axis (O2)</SectionTag>

# K-scaling — reconstructed vorticity

<img :src="'/images/kscaling_vorticity_comparison.png'" class="mt-3" style="display:block; width:100%; max-width:94%; margin:0 auto; max-height:300px; object-fit:contain;" />

<div class="foot mt-2" style="text-align:center;">Reconstructed vorticity <span class="raw">ω</span> at <span class="raw">t</span> = 5, single seed at the final protocol; each row shares one colour scale.</div>

<FooterLogos />

<!--
[K-scaling 渦度場 · 1min]
• 場層級：加 sensor 只讓小尺度**稍微銳利**，不是戲劇性改善
⚠️ ω rel-L₂ 38→31→30 %（K=200→400 幾乎飽和），與能譜頁 KE 5.90→2.47→1.76 % 不同量級：
   KE 低波數主導（大幅改善）、渦度高波數（受限）。這正是「中高頻受 sensor 上限限制」
• 講法：「能譜頁看到 cutoff 右移，這頁看到它在渦度場上長什麼樣 —— 大結構都對，高剪切核心最難」
⚠️ 圖上不標各欄 rel-L₂：single-seed 單幀
-->

---

<NavBar active="results" />

<SectionTag>§ Results · cross-Reynolds feasibility</SectionTag>

# Cross-Reynolds feasibility

<style>
.cre { width:100%; border-collapse:collapse; margin-top:24px; font-size:1.08rem;
       font-variant-numeric:tabular-nums; }
.cre th { text-align:right; font-weight:700; color:#374151; font-size:0.94rem;
          padding:0 18px 13px; border-bottom:1.5px solid #1F1B2E; white-space:nowrap; }
.cre th:first-child { text-align:left; }
.cre td { padding:17px 18px; border-bottom:1px solid #E5E0EC; color:#374151; text-align:right; }
.cre td:first-child { text-align:left; color:#1F1B2E; font-weight:600; }
.cre tr:last-child td { border-bottom:1.5px solid #1F1B2E; }
.cre tr.main td { background:rgba(127,16,132,0.07); }
.cre tr.main td:first-child { color:#7F1084; }
.cre .pend { color:#9CA3AF; font-style:italic; }
.cre .sub { font-weight:400; color:#9CA3AF; font-size:0.86em; }
</style>

<table class="cre">
<thead>
<tr>
<th>Reynolds number</th>
<th>Sensors <span class="raw">K</span></th>
<th>Seeds</th>
<th>KE MAPE (%)</th>
<th><span class="raw">u</span> rel-L₂ (%)</th>
<th><span class="raw">v</span> rel-L₂ (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td><span class="raw">Re</span> = 10³</td>
<td>100</td>
<td>1</td>
<td>2.36</td>
<td>5.79</td>
<td>5.07</td>
</tr>
<tr class="main">
<td><span class="raw">Re</span> = 10⁴ <span class="sub">main pipeline</span></td>
<td>100</td>
<td>5</td>
<td>5.71 ± 0.12</td>
<td>13.65</td>
<td>17.52</td>
</tr>
<tr>
<td><span class="raw">Re</span> = 10⁶</td>
<td>200</td>
<td>1</td>
<td>6.10</td>
<td>15.62</td>
<td>58.17</td>
</tr>
</tbody>
</table>

<div class="foot mt-4">Feasibility across two orders of magnitude in Reynolds number, not a controlled scaling study: the sensor budget and seed count differ by row. <span class="raw">Re</span> = 10⁶ is single-seed with a retuned configuration (<span class="raw">K</span> = 200, <span class="raw">d</span> = 384, 50 000 steps).</div>

<FooterLogos />

<!--
[Cross-Re · 1.5min]
• 主訊息：**同一架構跨兩個數量級的 Re 都 feasible**，KE 皆 ~6%（10⁴ 5.71、10⁶ 6.10）
⚠️ 必講成 **feasibility 不是 generalisation**（chapter05:20）：每列的 K、seed、config 都不同，
   不是受控 scaling 研究。Re=10⁶ 是 single-seed、retuned（K=200, d=384, 50k）
• Re=10³ = EXP-301 final-protocol 重跑（KE 2.36 / u 5.79 / v 5.07，single seed；eval job 4552，final.pt）
  取代不可比的舊 EXP-230（9.61%，d=64/5k/無AL）→ 同 config 重跑後 2.36%，比 10⁴ 還低
  同 K=100 下：Re 10³→10⁴ 誤差 2.36→5.71%（越高越難）；10⁶ 需 K=200 才維持 ~6%
⚠️ Re=10⁶ 的 v rel-L₂ 58% 遠高於 u 15.6%：高 Re 下 u/v 不對稱大。被問就誠實答，不掩飾
   （數字出處 experiment_log_v2:208）。KE 才是本頁 headline，不是 pointwise
• 數字：10⁴ log:132、10⁶ log:208；Re=10³ 為 EXP-301 final-protocol seed 42 結果
-->

---

<NavBar active="results" />

<SectionTag>§ Results · sensor noise axis (O3)</SectionTag>

# Reconstruction accuracy under sensor noise

<style>
.nz { width: 100%; border-collapse: collapse; font-size: 0.90rem; margin-top: 14px; margin-bottom: 0;
      font-variant-numeric: tabular-nums; }
.nz th { text-align: right; font-weight: 700; color: #6B7280; font-size: 0.90rem; text-transform: uppercase;
         letter-spacing: 0.04em; padding: 0 12px 6px 12px; border-bottom: 1px solid #D8D2E0; white-space: nowrap; }
.nz th:first-child { text-align: left; }
.nz th.worst { color: #1F1B2E; }
.nz th.delta { color: #E97132; border-left: 1px solid #D8D2E0; }
.nz .raw { text-transform: none; }
.nz td { padding: 9px 12px; border-bottom: 1px solid #F1EDF5; color: #9CA3AF; text-align: right; }
.nz td.m { text-align: left; color: #1F1B2E; font-weight: 600; white-space: nowrap; }
.nz td.m .tgt { font-weight: 400; color: #9CA3AF; }
.nz td.worst { color: #1F1B2E; font-weight: 700; background: #FAFAFC; }
.nz td.delta { color: #E97132; font-weight: 700; border-left: 1px solid #E5E0EC; }
.nz tr.head td { color: #1F1B2E; }
.nz tr.head td.worst { color: #1F1B2E; background: #F7EDF8; }
</style>

<div class="text-xs mt-1" style="color:#6B7280;">
Additive Gaussian noise, per-channel, as a fraction of each sensor's standard deviation, <b>n = 5 seeds per level</b>, final protocol.
</div>

<table class="nz">
<thead>
<tr>
<th>Metric (%)</th>
<th>0 %</th>
<th>1 %</th>
<th>3 %</th>
<th>5 %</th>
<th class="worst">10 %</th>
<th class="delta">Δ <span class="raw">0 → 10 %</span></th>
</tr>
</thead>
<tbody>
<tr class="head">
<td class="m">KE MAPE &nbsp;<span class="tgt">target &lt; 10</span></td>
<td>5.71</td><td>5.75</td><td>5.81</td><td>5.92</td>
<td class="worst">6.08</td>
<td class="delta">+6.5 %</td>
</tr>
<tr>
<td class="m">u rel-L₂</td>
<td>13.65</td><td>13.66</td><td>13.74</td><td>13.90</td>
<td class="worst">14.49</td>
<td class="delta">+6.2 %</td>
</tr>
<tr>
<td class="m">v rel-L₂</td>
<td>17.52</td><td>17.57</td><td>17.70</td><td>17.92</td>
<td class="worst">18.77</td>
<td class="delta">+7.1 %</td>
</tr>
<tr>
<td class="m"><span class="raw">ω</span> rel-L₂</td>
<td>41.77</td><td>41.78</td><td>42.00</td><td>42.32</td>
<td class="worst">43.47</td>
<td class="delta">+4.0 %</td>
</tr>
<tr>
<td class="m">div ratio</td>
<td>0.39</td><td>0.40</td><td>0.40</td><td>0.42</td>
<td class="worst">0.46</td>
<td class="delta">+17.7 %</td>
</tr>
</tbody>
</table>

<div class="mt-4 text-sm" style="color:#374151;">
All metrics degrade monotonically in the aggregate; none exceeds the 10 % target. At the worst case tested, KE sits <b style="color:#7F1084;">3.9 percentage points</b> under target. <b>Reliability, not feasibility.</b>
</div>

<div class="mt-2 text-xs" style="color:#9CA3AF;">
KE seed spread ± 0.03–0.21 across levels; the 0 % → 1 % step is smaller than that spread.
</div>

<FooterLogos />

<!--
[噪音 · 1.5min]
• Δ 欄全正 → noise 不是免費的；div 動最多（+17.7 %）
• KE 10 % 格 6.08，離門檻還有 3.9 pp。**先承認再劃線**
⚠️ 0→1 % 小於 seed 散布，不可宣稱可分辨
⚠️ ω 非嚴格單調 → 用「monotone in the aggregate」
⚠️ 別用舊的 single-seed 表
-->


---

<NavBar active="results" />

<SectionTag>§ Results · sensor noise axis (O3)</SectionTag>

# What 10 % sensor noise changes: input → reconstruction

<img :src="'/images/noise_input_output_comparison.png'" class="mt-2" style="display:block; width:100%; max-width:100%; margin:0 auto; max-height:325px; object-fit:contain;" />

<div class="foot mt-2" style="text-align:center;">Left: equivalent full-field visualization of the 10 % noise level. The actual training protocol applies the same noise model only to K = 100 measured <span class="raw">u,v</span> values.</div>

<FooterLogos />

<!--
[噪音下的渦度場 · 1min]
• 左邊是教授要求的 full-field noise 視覺化：clean DNS u/v 與加入等效 10 % Gaussian noise 後的場
• 必須主動說這是 illustration；實際模型仍只看到 K=100 個 noisy sensor values，不吃 full field
• 10 % 指每個 channel 的 σ_noise = 0.1 σ_signal，實驗中是在 normalization 前注入
• 右邊：clean 與 10 % noisy input 的重建主結構接近；誤差仍集中在渦核與剪切層
⚠️ 這是 t=5、seed 42 的視覺例子；統計結論仍以前一頁 n=5 表格為準
-->

---

<NavBar active="results" />

<SectionTag>§ Results · engineering applicability</SectionTag>

# What K = 100 sensors support

<style>
/* 兩欄的寬度 = 波數帶的分割比例，截止線因此一路貫穿到底 —— 帶子是結構，不是裝飾。
   顏色兩個意思：紫 = 截止線以內（可解析）, 灰 = 此感測預算下約束不足。 */
.sp { display: grid; grid-template-columns: 41.6% 58.4%; }
.band { height: 34px; }
.band .lo { background: rgba(127,16,132,0.14); border: 1px solid #7F1084; border-right: 2px solid #E97132;
            border-radius: 5px 0 0 5px; display: flex; align-items: center; justify-content: center; }
.band .hi { background: repeating-linear-gradient(45deg, #F6F6F8, #F6F6F8 5px, #EBEBEF 5px, #EBEBEF 10px);
            border: 1px solid #D8D2E0; border-left: none; border-radius: 0 5px 5px 0;
            display: flex; align-items: center; justify-content: center; }
.band .lbl { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; }
.kx { font-size: 0.68rem; color: #9CA3AF; margin-top: 4px; }
.kx b { color: #E97132; }
.col { padding: 8px 14px 0 0; }
.col.r { padding-left: 16px; border-left: 2px solid #E97132; }
.col h4 { font-size: 0.9rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 9px; }
.col .row { font-size: 0.9rem; line-height: 1.5; color: #374151; }
.col .row b { font-weight: 700; }
.ar { color: #C9C6D0; font-weight: 400; }
</style>

<div class="sp band mt-3">
  <div class="lo"><span class="lbl" style="color:#7F1084;">RESOLVED, 98.9 % of the energy</span></div>
  <div class="hi"><span class="lbl" style="color:#9CA3AF;">POORLY CONSTRAINED</span></div>
</div>
<div class="sp kx">
  <span>k = 1</span>
  <span style="padding-left:6px;"><b>k<sub>max</sub> = √(K/π) = 5.64</b> <span class="ar">→</span> sensor Nyquist scale — a sensor budget, not an architecture</span>
</div>

<div class="sp">

<div class="col">
<h4 style="color:#7F1084;">Supported</h4>
<div class="row"><b style="color:#7F1084;">KE &amp; mean-flow monitoring</b><br/><span class="ar">→</span> 5.71 ± 0.12 %</div>
<div class="row mt-2"><b style="color:#7F1084;">Incompressibility check</b><br/><span class="ar">→</span> div 0.39 % = FD floor</div>
<div class="row mt-2"><b style="color:#7F1084;">Streaming deployment</b><br/><span class="ar">→</span> causal, queries at any t</div>
</div>

<div class="col r">
<h4 style="color:#E97132;">Out of scope</h4>
<div class="row"><b style="color:#E97132;">Small-scale statistics</b> <span class="ar">→</span> high-order moments beyond k<sub>max</sub></div>
<div class="row mt-2"><b style="color:#E97132;">Fine vorticity filaments</b> <span class="ar">→</span> ω is a diagnostic, not an observable</div>
<div class="row mt-2"><b style="color:#E97132;">Acoustic / shock localisation</b> <span class="ar">→</span> needs denser or multi-modal sensing</div>
<div class="row mt-3" style="color:#6B7280;">The fix is <b>more sensors</b>, not a bigger network.</div>
</div>

</div>


<FooterLogos />

<!--
[工程適用範圍 · 2min]
• 左＝支援：KE 與 mean-flow 監測／不可壓縮檢查／streaming
• 右＝不支援：小尺度統計／細渦絲／聲學定位
• 收尾：加 sensor，不是加大網路
-->

---

<NavBar active="summary" />

<SectionTag>§ Conclusion · contributions</SectionTag>

# Contributions

<style>
.ct { display: grid; grid-template-columns: max-content 1fr; column-gap: 13px; row-gap: 15px; margin-top: 20px; }
.ct .num { font-size: 1.02rem; font-weight: 700; color: #7F1084; line-height: 1.45; }
.ct .body { font-size: 1.02rem; line-height: 1.45; color: #374151; }
.ct .body b { color: #7F1084; }
.ct .body i { font-style: normal; color: #9CA3AF; }
.ar { color: #9CA3AF; font-weight: 400; }
</style>

<div class="ct">

<div class="num">1.</div>
<div class="body">
Developed <b>PI-CON</b>, trained only on sensors and the NS residual; matched-budget ablation improves KE error from 8.23 % to <b>5.71 %</b> <i>(p = 3 × 10⁻⁷)</i>.
</div>

<div class="num">2.</div>
<div class="body">
Reconstructed Re = 10⁴ Kolmogorov flow from <b>K = 100</b> sensors to <b>5.71 ± 0.12 %</b> KE error over five seeds, without DNS full-field supervision.
</div>

<div class="num">3.</div>
<div class="body">
Verified count-limited bandwidth: at K = 100, effective cutoffs <b>4.7–7.8</b> bracket √(K/π) = 5.64; increasing K from 100 to 400 reduces KE error by <b>70 %</b>.
</div>

<div class="num">4.</div>
<div class="body">
Quantified reliability: placement variability is <b>6.4×</b> training variability, while 10 % sensor noise remains below the target <i>(6.08 % KE error)</i>.
</div>

<div class="num">5.</div>
<div class="body">
Conducted a <b>cross-Reynolds feasibility check</b> beyond the primary Re = 10⁴ setting <i>(a retuned configuration, not a controlled scaling study)</i>.
</div>

</div>

<FooterLogos />

<!--
[貢獻 · 2min]
• 逐條回扣 O1–O3：精度與架構增益／sensor-count bandwidth／placement 與 noise；最後補跨 Re feasibility
• Cross-Re 是額外 feasibility extension；只說確認過，不把它包裝成 controlled scaling 或 generalisation
⚠️ 數字出處：log:132／399–419／580／1561；chapter04:324／494–496
-->
