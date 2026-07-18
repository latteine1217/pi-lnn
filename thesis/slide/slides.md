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
[Cover · 30s] PI-CON 論文 defense。重點 anchor 在標題那行：K=100 sensors only · NS residual as the only physics signal · no DNS supervision in training。大綱 → 問題 / 架構 / 訓練 / 結果（能力→數量→位置→噪音三軸）/ 限制 / 下一步。
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
[Why reconstruction · 1min] 2026-07-18 依指導教授 meeting 新增。
教授原話：「說明為什麼要做 reconstruction 問題 —— 因為真實情況只有 sensor，沒有全場資料，
所以需要 reconstruction 來重建。」

本頁只建立一般背景：儀器通常只能量到局部，但分析與工程決策需要整個空間中的結構、梯度與
整體量。因此需要 reconstruction 作為「局部觀測到連續場」之間的工具。不要在本頁提 K、DNS、
Navier–Stokes 或本研究架構；那些條件留到 literature review 後的 problem formulation。
-->

---

<NavBar active="background" />

<SectionTag>§ Background · what a PINN does</SectionTag>

# Physics-informed neural networks

<div class="text-xs mt-1" style="color:#374151;">
The network <b>is</b> the field: training adjusts θ from data and physics; inference reuses the frozen θ in one forward pass.
</div>

<div class="mt-1">
<svg class="pinn-old" viewBox="0 0 900 372" style="width:100%;height:auto;max-height:344px;">

  <!-- ============ panel ① TRAINING ============ -->
  <rect x="10" y="28" width="880" height="246" rx="8" fill="#FCFCFD" stroke="#D8D2E0" stroke-width="1.2" stroke-dasharray="6 4"/>
  <text x="24" y="20" fill="#0F2D52" style="font-size:11.5px;font-weight:700;letter-spacing:0.06em;">① TRAINING — fit the weights θ</text>

  <!-- network edges -->
  <g stroke="#0F2D52" stroke-width="0.6" opacity="0.22">
    <line x1="112" y1="122" x2="150" y2="76"/>
    <line x1="112" y1="122" x2="150" y2="104"/>
    <line x1="112" y1="122" x2="150" y2="132"/>
    <line x1="112" y1="122" x2="150" y2="160"/>
    <line x1="150" y1="76" x2="205" y2="76"/>
    <line x1="150" y1="76" x2="205" y2="104"/>
    <line x1="150" y1="76" x2="205" y2="132"/>
    <line x1="150" y1="76" x2="205" y2="160"/>
    <line x1="150" y1="104" x2="205" y2="76"/>
    <line x1="150" y1="104" x2="205" y2="104"/>
    <line x1="150" y1="104" x2="205" y2="132"/>
    <line x1="150" y1="104" x2="205" y2="160"/>
    <line x1="150" y1="132" x2="205" y2="76"/>
    <line x1="150" y1="132" x2="205" y2="104"/>
    <line x1="150" y1="132" x2="205" y2="132"/>
    <line x1="150" y1="132" x2="205" y2="160"/>
    <line x1="150" y1="160" x2="205" y2="76"/>
    <line x1="150" y1="160" x2="205" y2="104"/>
    <line x1="150" y1="160" x2="205" y2="132"/>
    <line x1="150" y1="160" x2="205" y2="160"/>
    <line x1="205" y1="76" x2="260" y2="76"/>
    <line x1="205" y1="76" x2="260" y2="104"/>
    <line x1="205" y1="76" x2="260" y2="132"/>
    <line x1="205" y1="76" x2="260" y2="160"/>
    <line x1="205" y1="104" x2="260" y2="76"/>
    <line x1="205" y1="104" x2="260" y2="104"/>
    <line x1="205" y1="104" x2="260" y2="132"/>
    <line x1="205" y1="104" x2="260" y2="160"/>
    <line x1="205" y1="132" x2="260" y2="76"/>
    <line x1="205" y1="132" x2="260" y2="104"/>
    <line x1="205" y1="132" x2="260" y2="132"/>
    <line x1="205" y1="132" x2="260" y2="160"/>
    <line x1="205" y1="160" x2="260" y2="76"/>
    <line x1="205" y1="160" x2="260" y2="104"/>
    <line x1="205" y1="160" x2="260" y2="132"/>
    <line x1="205" y1="160" x2="260" y2="160"/>
    <line x1="260" y1="76" x2="298" y2="122"/>
    <line x1="260" y1="104" x2="298" y2="122"/>
    <line x1="260" y1="132" x2="298" y2="122"/>
    <line x1="260" y1="160" x2="298" y2="122"/>
  </g>
  <g fill="#0F2D52">
    <circle cx="150" cy="76" r="5.5"/>
    <circle cx="150" cy="104" r="5.5"/>
    <circle cx="150" cy="132" r="5.5"/>
    <circle cx="150" cy="160" r="5.5"/>
    <circle cx="205" cy="76" r="5.5"/>
    <circle cx="205" cy="104" r="5.5"/>
    <circle cx="205" cy="132" r="5.5"/>
    <circle cx="205" cy="160" r="5.5"/>
    <circle cx="260" cy="76" r="5.5"/>
    <circle cx="260" cy="104" r="5.5"/>
    <circle cx="260" cy="132" r="5.5"/>
    <circle cx="260" cy="160" r="5.5"/>
  </g>

  <!-- neuron-count annotation with a span bracket -->
  <path d="M120 62 L120 54 L290 54 L290 62" fill="none" stroke="#9CA3AF" stroke-width="1"/>
  <text x="205" y="46" text-anchor="middle" fill="#0F2D52" style="font-size:11.5px;font-weight:700;">n layers × m neurons</text>
  <text x="250" y="200" text-anchor="middle" fill="#9CA3AF" style="font-size:11.5px;font-weight:400;">N(x, y, t ; θ)</text>

  <!-- inputs -->
  <circle cx="58" cy="88" r="14" fill="#FFF" stroke="#0F2D52" stroke-width="1.6"/>
  <text x="58" y="88" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:13px;font-weight:700;font-style:italic;">x</text>
  <circle cx="58" cy="122" r="14" fill="#FFF" stroke="#0F2D52" stroke-width="1.6"/>
  <text x="58" y="122" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:13px;font-weight:700;font-style:italic;">y</text>
  <circle cx="58" cy="156" r="14" fill="#FFF" stroke="#0F2D52" stroke-width="1.6"/>
  <text x="58" y="156" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:13px;font-weight:700;font-style:italic;">t</text>

  <!-- outputs -->
  <circle cx="340" cy="88" r="14" fill="#FFF" stroke="#0F2D52" stroke-width="1.6"/>
  <text x="340" y="88" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:13px;font-weight:700;font-style:italic;">u</text>
  <circle cx="340" cy="122" r="14" fill="#FFF" stroke="#0F2D52" stroke-width="1.6"/>
  <text x="340" y="122" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:13px;font-weight:700;font-style:italic;">v</text>
  <circle cx="340" cy="156" r="14" fill="#FFF" stroke="#0F2D52" stroke-width="1.6"/>
  <text x="340" y="156" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:13px;font-weight:700;font-style:italic;">p</text>

  <!-- outputs -> autodiff -->
  <line x1="356" y1="96" x2="390" y2="80" stroke="#7F1084" stroke-width="1.5"/>
  <path d="M397 77 L389 77 L392 84 Z" fill="#7F1084"/>
  <rect x="400" y="48" width="140" height="56" rx="5" fill="#FAF2FB" stroke="#7F1084" stroke-width="1.5"/>
  <text x="470" y="66" text-anchor="middle" dominant-baseline="central" fill="#7F1084" style="font-size:10.5px;font-weight:700;letter-spacing:0.05em;">AUTODIFF</text>
  <text x="470" y="88" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:12px;font-weight:400;">∂u/∂t,  ∇u,  ∇²u</text>

  <!-- autodiff -> PDE residual -->
  <line x1="540" y1="76" x2="560" y2="76" stroke="#E97132" stroke-width="1.5"/>
  <path d="M567 76 L559 72.5 L559 79.5 Z" fill="#E97132"/>
  <rect x="570" y="44" width="200" height="64" rx="5" fill="#FEF6F1" stroke="#E97132" stroke-width="1.5"/>
  <text x="670" y="59" text-anchor="middle" dominant-baseline="central" fill="#E97132" style="font-size:10.5px;font-weight:700;letter-spacing:0.05em;">PDE RESIDUAL</text>
  <text x="670" y="78" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:10.5px;font-weight:400;">∂u/∂t + (u·∇)u + ∇p − ν∇²u</text>
  <text x="670" y="96" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:11.5px;font-weight:400;">∇·u = 0</text>

  <!-- outputs -> sensor misfit (orthogonal routing, no crossings) -->
  <path d="M356 150 L370 150 Q378 150 378 158 L378 177 Q378 185 386 185 L560 185" fill="none" stroke="#E97132" stroke-width="1.5"/>
  <path d="M567 185 L559 181.5 L559 188.5 Z" fill="#E97132"/>
  <rect x="570" y="160" width="200" height="50" rx="5" fill="#FEF6F1" stroke="#E97132" stroke-width="1.5"/>
  <text x="670" y="174" text-anchor="middle" dominant-baseline="central" fill="#E97132" style="font-size:10.5px;font-weight:700;letter-spacing:0.05em;">SENSOR MISFIT</text>
  <text x="670" y="195" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:11.5px;font-weight:400;">u(x&#8342;, t) − u&#8342;&#7506;&#7495;&#738;</text>

  <!-- residuals -> total loss -->
  <line x1="770" y1="76" x2="789" y2="114" stroke="#9CA3AF" stroke-width="1.4"/>
  <path d="M795 121 L787 118 L793 113 Z" fill="#9CA3AF"/>
  <line x1="770" y1="185" x2="789" y2="147" stroke="#9CA3AF" stroke-width="1.4"/>
  <path d="M795 140 L793 148 L787 143 Z" fill="#9CA3AF"/>
  <rect x="800" y="104" width="78" height="54" rx="5" fill="#F4F0F7" stroke="#7F1084" stroke-width="1.5"/>
  <text x="839" y="122" text-anchor="middle" dominant-baseline="central" fill="#7F1084" style="font-size:15px;font-weight:700;font-style:italic;">L(θ)</text>
  <text x="839" y="142" text-anchor="middle" dominant-baseline="central" fill="#9CA3AF" style="font-size:9.5px;font-weight:400;">weighted sum</text>

  <!-- loss -> optimizer -->
  <path d="M839 158 L839 233 Q839 241 831 241 L768 241" fill="none" stroke="#7F1084" stroke-width="1.5"/>
  <path d="M761 241 L769 237.5 L769 244.5 Z" fill="#7F1084"/>
  <rect x="556" y="224" width="200" height="34" rx="5" fill="#F4F0F7" stroke="#7F1084" stroke-width="1.5"/>
  <text x="656" y="241" text-anchor="middle" dominant-baseline="central" fill="#7F1084" style="font-size:11.5px;font-weight:700;">OPTIMIZER &#8201;·&#8201; gradient descent</text>

  <!-- optimizer -> back into the network -->
  <path d="M556 241 L128 241 Q120 241 120 233 L120 192" fill="none" stroke="#7F1084" stroke-width="1.5"/>
  <path d="M120 185 L116.5 193 L123.5 193 Z" fill="#7F1084"/>
  <text x="340" y="230" text-anchor="middle" fill="#7F1084" style="font-size:11px;font-weight:700;">update θ by back-propagation</text>

  <!-- ============ panel ② INFERENCE ============ -->
  <rect x="10" y="300" width="880" height="62" rx="8" fill="#FAFAFC" stroke="#0F2D52" stroke-width="1.2"/>
  <text x="24" y="292" fill="#0F2D52" style="font-size:11.5px;font-weight:700;letter-spacing:0.06em;">② INFERENCE — θ frozen</text>

  <rect x="30" y="313" width="132" height="36" rx="5" fill="#FFF" stroke="#0F2D52" stroke-width="1.3"/>
  <text x="96" y="331" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:11.5px;font-weight:400;">any (x, y, t)</text>
  <line x1="162" y1="331" x2="182" y2="331" stroke="#9CA3AF" stroke-width="1.3"/>
  <path d="M189 331 L181 327.5 L181 334.5 Z" fill="#9CA3AF"/>
  <rect x="192" y="313" width="140" height="36" rx="5" fill="#F4F6F9" stroke="#0F2D52" stroke-width="1.3"/>
  <text x="262" y="331" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:11.5px;font-weight:700;">N( · ; θ*)</text>
  <line x1="332" y1="331" x2="352" y2="331" stroke="#9CA3AF" stroke-width="1.3"/>
  <path d="M359 331 L351 327.5 L351 334.5 Z" fill="#9CA3AF"/>
  <rect x="362" y="313" width="106" height="36" rx="5" fill="#FFF" stroke="#0F2D52" stroke-width="1.3"/>
  <text x="415" y="331" text-anchor="middle" dominant-baseline="central" fill="#0F2D52" style="font-size:11.5px;font-weight:400;">u, v, p</text>
  <text x="492" y="331" dominant-baseline="central" fill="#6B7280" style="font-size:11.5px;font-weight:400;">one forward pass &#8201;·&#8201; no mesh, no time-stepping, no solver in the loop</text>

</svg>
</div>

<style>
.pinn-old { display:none; }
.pinn { display:none; }
.pinn .panel { border:1px solid #DDD6E5; border-radius:11px; padding:11px 14px; background:rgba(255,255,255,.76); }
.pinn .head { display:flex; align-items:baseline; gap:10px; margin-bottom:8px; }
.pinn .tag { font-size:.74rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }
.pinn .desc { font-size:.76rem; color:#6B7280; }
.pinn .train-flow { display:grid; grid-template-columns:1fr 22px 1.18fr 22px 1fr 22px 1.25fr 22px .9fr; align-items:center; }
.pinn .infer-flow { display:grid; grid-template-columns:1fr 32px 1.3fr 32px 1fr; align-items:center; }
.pinn .box { border:1px solid #D8D2E0; border-radius:7px; padding:8px 7px; min-height:56px; display:flex; flex-direction:column; justify-content:center; text-align:center; background:#fff; }
.pinn .box .small { font-size:.70rem; color:#6B7280; line-height:1.22; margin-top:2px; }
.pinn .box .main { font-size:.86rem; font-weight:700; color:#1F1B2E; line-height:1.22; }
.pinn .net { position:relative; background:#F4F6F9; border-color:#0F2D52; }
.pinn .net::before { content:'n layers × m neurons'; position:absolute; top:-17px; left:0; right:0; font-size:.67rem; font-weight:700; color:#0F2D52; }
.pinn .arr { text-align:center; color:#7F1084; font-size:1.2rem; font-weight:700; }
.pinn .loss-stack { display:grid; grid-template-rows:1fr 1fr; gap:5px; }
.pinn .loss { border-radius:6px; padding:5px 7px; border:1px solid #E9C9B2; background:#FEF6F1; text-align:left; }
.pinn .loss b { display:block; font-size:.68rem; letter-spacing:.04em; color:#E97132; }
.pinn .loss span { font-size:.68rem; color:#374151; line-height:1.15; }
.pinn .optimizer { background:#FAF2FB; border-color:#7F1084; }
.pinn .feedback { margin:7px 10% 0 15%; border-top:1.5px solid #7F1084; position:relative; text-align:center; color:#7F1084; font-size:.70rem; font-weight:700; padding-top:3px; }
.pinn .feedback::before { content:'↖'; position:absolute; left:-11px; top:-12px; font-size:1rem; }
.pinn .frozen { background:#FAF2FB; border-color:#7F1084; }
.pinn > .panel:last-child { padding:8px 14px; }
.pinn > .panel:last-child .head { margin-bottom:4px; }
.pinn > .panel:last-child .box { min-height:46px; padding:6px 7px; }
.pinn .inference-note { text-align:center; font-size:.70rem; color:#6B7280; margin-top:2px; }
</style>

<div class="pinn">
  <div class="panel">
    <div class="head"><div class="tag" style="color:#7F1084;">01 Training</div><div class="desc">evaluate two residuals, then update the network weights</div></div>
    <div class="train-flow">
      <div class="box"><div class="main">Coordinates</div><div class="small">(x, y, t)</div></div><div class="arr">→</div>
      <div class="box net"><div class="main">Neural field N(·; θ)</div><div class="small">trainable weights θ</div></div><div class="arr">→</div>
      <div class="box"><div class="main">Prediction</div><div class="small">u, v, p</div></div><div class="arr">→</div>
      <div class="loss-stack"><div class="loss"><b>DATA RESIDUAL</b><span>compare at measured points</span></div><div class="loss"><b>PDE RESIDUAL</b><span>autodiff → governing equations</span></div></div><div class="arr">→</div>
      <div class="box optimizer"><div class="main" style="color:#7F1084;">Loss → optimizer</div><div class="small">back-propagation</div></div>
    </div>
    <div class="feedback">optimizer changes θ; the cycle repeats until the residuals are minimized</div>
  </div>
  <div class="panel">
    <div class="head"><div class="tag" style="color:#0F2D52;">02 Inference</div><div class="desc">freeze θ* and evaluate the learned field directly</div></div>
    <div class="infer-flow">
      <div class="box"><div class="main">Query coordinate</div><div class="small">any (x, y, t)</div></div><div class="arr">→</div>
      <div class="box frozen"><div class="main" style="color:#7F1084;">Trained field N(·; θ*)</div><div class="small">one forward pass</div></div><div class="arr">→</div>
      <div class="box"><div class="main">Field value</div><div class="small">u, v, p</div></div>
    </div>
    <div class="inference-note">No loss evaluation, optimizer, or back-propagation</div>
  </div>
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
[What a PINN does · 1min] 2026-07-18 依指導教授 meeting 新增。
教授原話：「加一頁說明 PINNs 的功能，以及如何運作。」

口述順序依上下兩張 card：
**① Training** —— 座標送入 neural field 得到預測；預測同時與量測形成 data residual，並由
autodiff 代入統御方程形成 PDE residual。兩者合成 loss，optimizer 透過 back-propagation 更新 θ。
**② Inference** —— θ* 凍結後，只需把查詢座標送入已訓練的 neural field，直接得到場值；
不再計算 loss，也沒有 optimizer 或 back-propagation。

⚠️ 2D 形式（x, y, t → u, v, p），與本研究的 Kolmogorov case 一致；
教授提供的參考圖為 3D（含 z, w），已依其確認改為 2D。

⚠️ 本頁是**通論**，講的是 PINN 一般形式，不是 PI-CON。
PI-CON 與 vanilla PINN 的差別（sensor 讀進網路 vs 只進 loss）在 Motivation 段的
「Operator vs. plain PINN」頁才展開，本頁不要提前講，否則兩頁重複。

圖中的 n layers × m neurons 是一般神經網路表示，不對應本研究的實際層數與寬度。
-->

---

<NavBar active="literature" />

<SectionTag>§ Literature review · classical inverse methods</SectionTag>

# What classical methods require

<style>
.cl { width: 100%; border-collapse: collapse; font-size: 0.92rem; margin-top: 16px; }
.cl th { text-align: left; font-weight: 700; color: #9CA3AF; font-size: 0.68rem; text-transform: uppercase;
         letter-spacing: 0.05em; padding: 0 10px 6px 10px; border-bottom: 1px solid #D8D2E0; }
.cl th.x { text-align: center; }
.cl td { padding: 9px 10px; border-bottom: 1px solid #F1EDF5; color: #374151; vertical-align: middle; }
.cl td.who { color: #1F1B2E; font-weight: 600; white-space: nowrap; }
.cl td.src { color: #9CA3AF; font-size: 0.82em; }
.cl td.x { text-align: center; font-weight: 700; font-size: 1.05rem; }
.cl .no { color: #E97132; }
.cl tr.grp td { border-bottom: none; padding-top: 13px; padding-bottom: 2px;
                font-size: 0.71rem; font-weight: 700; letter-spacing: 0.05em;
                text-transform: uppercase; color: #1F1B2E; }
</style>

<table class="cl">
<thead>
<tr>
<th style="width: 20%;">Method</th>
<th style="width: 34%;">Source</th>
<th style="width: 23%;" class="x">Offline field record</th>
<th style="width: 23%;" class="x">Solver in the loop</th>
</tr>
</thead>
<tbody>

<tr class="grp"><td colspan="4">Reduced-order models</td></tr>
<tr><td class="who">POD</td><td class="src">Sirovich 1987, Q. Appl. Math.</td><td class="x no">required</td><td class="x">—</td></tr>
<tr><td class="who">DMD</td><td class="src">Schmid 2010, J. Fluid Mech.</td><td class="x no">required</td><td class="x">—</td></tr>
<tr><td class="who">QR-pivot</td><td class="src">Manohar 2018, IEEE Control Syst. Mag.</td><td class="x no">required</td><td class="x">—</td></tr>

<tr class="grp"><td colspan="4">Data assimilation</td></tr>
<tr><td class="who">4D-Var</td><td class="src">Asch 2016, SIAM</td><td class="x">—</td><td class="x no">required</td></tr>
<tr><td class="who">EnKF</td><td class="src">Asch 2016, SIAM</td><td class="x">—</td><td class="x no">required</td></tr>

</tbody>
</table>

<div class="mt-4" style="display:grid; grid-template-columns:max-content 1fr; column-gap:14px; row-gap:5px; align-items:baseline; font-size:0.90rem; border-left:2px solid #E97132; padding-left:12px;">
<span style="color:#9CA3AF; white-space:nowrap;">On a rig</span><span style="color:#374151;">No field record before it runs <span style="color:#C9C6D0;">, </span> a solver in the loop costs minutes to hours per window at Re = 10⁴</span>
</div>

<FooterLogos />

<!--
[Classical methods · 1.5min] 2026-07-18 依指導教授 meeting 改版：
原為兩張說明卡，改為與其餘 literature review 頁一致的比較表，並移入 Literature 段。

**在 literature review 中的位置**：本頁排在最前，順序是「傳統 → 學習式 → 感測器如何進入」，
由舊至新、由簡至繁：
  1. 本頁              傳統方法要什麼（現場給不起）
  2. What prior methods are trained against   學習式方法對著什麼擬合（全都要全場）
  3. How the sensor stream enters the model   四種能力沒有一種方法全有

口述：「這兩類是流體重建的傳統做法。降階模型要先有一段完整流場的紀錄才能建基底；
資料同化則要把求解器放進迴圈裡，每個同化窗都重跑一次。第一項在現場不存在 ——
機台開始運轉前沒有全場紀錄；第二項在 Re=10⁴ 下每個窗要幾分鐘到幾小時，來不及。
所以先驗只能從稀疏感測器加上統御方程本身學出來。」

依據 chapter01.tex Table 1.1（:19-28）原文：
- ROM 列：「Compress the field onto a low-rank basis identified offline」，
  supervision「Offline DNS trajectory」，limitation「Needs offline trajectory; linear basis」
- DA 列：「Bayesian state estimation against a forward solver and noise model」，
  supervision「Forward solver + sensor」，limitation「Adjoint cost; HMC scaling at high Re」

⚠️ 未列 RBF / IDW / 三角最小平方等直接內插法：那三者在 appendix07 是本研究自建的
fair baseline、**無文獻 citation**，放進 literature review 會變成「跟通用方法比」而非文獻對照。
它們的比較在 Results 段的「Lower KE does not mean a better field」頁。

⚠️ 4D-Var 未實作為 baseline（見 Limitations 頁）。本頁只陳述其結構需求，
不宣稱數值上勝過它。委員若問「有沒有實際跑 4D-Var 比較」→ 誠實答：沒有，列為 future work。
-->
---

<NavBar active="literature" />

<SectionTag>§ Literature review · training supervision in prior work</SectionTag>

# What prior methods are trained against

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

<div class="dns-summary"><b>Comparison:</b> the architectures differ, but every method learns against a dense reference field. That supervision is the common deployment bottleneck.</div>


<FooterLogos />

<!--
[Literature review 2/3 · 1.5min] 這頁只有一個論點：它們全都對著 reference field 擬合。
故只有一個欄位帶色（橘＝loss 對著什麼擬合），其餘中性。前一版 7 種文字顏色、橘色同時
用在 supervision / Re / readout 三種意思上 —— 每格都是重點就等於沒有重點。

已移除 Readout 與 Sensors 欄：readout 是 slide 7 的軸（Parfenyev 的 query-anywhere 在那裡
才有意義）；sensor 數在此頁不承擔論點。Re 併入 Case 欄，未報者留白（—），不特別標色 ——
那是缺席，不是警訊。

⚠️⚠️ 口試發言警告（2026-07-18 查證後新增）：**不可說「我的 Re 比所有文獻都高」。**
SHRED 用的 JHTDB forced isotropic turbulence 資料集規格公開：**integral scale Re = 23,298**
（Re_λ = 433, 1024³, ν=1.85e-4；來源 turbulence.pha.jhu.edu 官方 dataset 頁）——
**是我們 Re=10⁴ 的 2.3 倍**，且為 3D isotropic 真湍流（有完整慣性range），
而我們是 2D Kolmogorov、Re_f≈2.5×10³、能譜斜率 −4.61、無慣性range。
流體領域委員很可能認得 JHTDB，講錯會被當場反駁並連累其他 claim 的可信度。

**正確說法（把 SHRED 的高 Re 轉成我們的論據）**：
「在**不使用全場監督**的同類方法裡，我們的 Re 最高（slide 7：Mo & Magri 34、Kelshaw 34、
Parfenyev 1.3×10³、我們 10⁴，7.7×）。至於 Re 更高的 SHRED（2.3×10⁴），它需要完整流場
當監督訊號 —— 那正是本研究不需要的東西。」
→ Re 的 head-to-head 只在 slide 7（same regime）成立；本頁的論點是**監督訊號**不是 Re。

逐格出處（2026-07-15 查證；Re 欄 2026-07-18 補）：
- SHRED (arXiv 2301.12011): stacked LSTM + shallow FC decoder；loss 原文
  「minimize reconstruction loss ∑ᵢ‖xᵢ − H̃({yⱼ})‖₂」→ 對全場 state 監督。
  流場原文：「the pressure field of a forced isotropic turbulent flow from the Johns Hopkins
  Turbulence Database」「generated by direct numerical simulation using 1024³ nodes」。
  **論文本身未報 Re**；表格填的 2.3×10⁴ 是該 JHTDB 資料集的公開規格，非論文所報 ——
  被問來源時要如此區分。
- Senseiver (Nature MI 2023): Perceiver IO 系 cross-attention 編碼進 latent；OSTI 摘要原文
  「a dense set of observations is needed to train」；Re / sensor 數未報；正文付費牆。
- FLRNet (arXiv 2411.13815): conv VAE + Gaussian Fourier (m=4, σ=5) + MLP 5×128；
  loss = VAE reconstruction + perceptual；cylinder Re 300–1000。
- FLRONet (arXiv 2412.08009): FNO branch (d=64) + 3-layer MLP trunk；cylinder CFDBench。
  ⚠️ 其 loss 定義原文未明述，「Paired CFD fields」依 chapter01:101 論述填入，非原文直引。

== 「FLRONet 的 Re 為什麼空著？」（2026-07-17 親自查證 arXiv 2412.08009 全文 15 頁）==
**因為原文從頭到尾沒給 Re 數值。** 實測：全文 54k 字元中「Reynolds」只出現 1 次，且無數字 ——
唯一那句是定性的「…the inherent difficulty of reconstructing flow with a high Reynolds number
driven by the increased velocity of the fluid…」。「viscosity」出現 0 次，所以連反推 Re 都做不到。

它改用**入口速度**索引案例：CFDBench cylinder dataset，domain 0.14 × 0.24 m → 140×240 grid，
50 個 case，inlet velocity 由 0.1 m/s 遞增到 5.0 m/s（45 train / 5 test，test 為 3.5/3.9/4.2…）。
⚠️ 2026-07-17：曾把「no Re stated · indexed by inlet velocity 0.1–5.0 m/s」以小字標進 Case 欄，
已移除（使用者：小字不需要）。**該資訊改為口述** —— 被問「FLRONet 的 Re 是多少」時答：
「原文沒給。全文只出現一次 Reynolds 且無數值，viscosity 一次都沒有，所以連反推都做不到；
它用 inlet velocity 0.1–5.0 m/s 索引 50 個 case。」
同理 SHRED (JHTDB isotropic) 與 Senseiver 亦未報 Re。**四篇裡三篇未報 Re，只有 FLRNet
給了 Re 300–10³** —— 這點若被問「為何不做 Re 的 head-to-head」，就是答案。

⚠️ 先前這裡的註記「Re 未報」是別人 2026-07-15 記的，我 2026-07-17 親自抓 PDF 逐字查證後確認屬實
（第一次 WebFetch 讀 PDF 回傳二進位亂碼、宣稱「無法確定」，不可採信；改用 pypdf 本地解析才得出）。
⚠️ 舊版頁面底部曾有一行小字「the Reynolds number is unstated in three of the four above」，
已被移除 —— 該資訊現在改標在各自的格子裡，比一行看不見的註腳有效。

順帶：FLRONet 論文標題即為「**Deep Operator Learning** for High-Fidelity Fluid Flow Field
Reconstruction from Sparse Sensor Measurements」，自稱 deep operator learning —— 這是本表
必須標出 DeepONet 血緣的第二個依據（第一個是 chapter01:101）。

== 「為什麼沒有 DeepONet 系的對照？」（2026-07-17 補；委員極可能問）==
有 —— 就是 FLRONet。chapter01:101 原文稱它是「the spatio-temporal **deep operator network**
of Vo Dang and Nguyen … **the closest published architecture to the present branch--trunk
readout**」。先前本表把它的 Architecture 寫成「FNO branch + MLP trunk」，沒出現 DeepONet
字樣，等於把這個對照藏起來 —— 委員讀表時會問的正是這題。已改為
「DeepONet · FNO branch + MLP trunk」+ 灰字註「closest published branch–trunk」。

完整答法（三層，被追問時逐層給）：
1. **文獻對照 = FLRONet**：同為 branch–trunk deep operator network，但它訓練對著
   paired CFD fields（chapter01:101「train against complete CFD fields rather than the PDE」）
   —— 差別不在架構家族，在監督訊號。這正是本表的論點。
2. **原版 DeepONet [Lu 2021] 為何不在同 regime 比較**：chapter01:93 明載 DeepONet/FNO
   「are demonstrated as dense-input forward operators: the branch expects its input function
   sampled on a fixed dense grid … rather than ~10² scattered points」——它吃不了稀疏散點，
   屬 Gap 2，不是同 regime 的競爭者。連 physics-informed 的 PINO 也「evaluates its residual
   on a grid」。
3. **真正的 DeepONet 對照在內部**：chapter01:128 的 O1 criterion 就是「reduce KE error by at
   least two percentage points relative to the **vanilla DeepONet baseline** at p<0.01」——
   那是 2×2 ablation 的 B0（主結果頁：B0 8.23% → B3 5.71%，−2.53 pp, p=3.0×10⁻⁷）。
   即 vanilla DeepONet 是以 baseline 而非文獻列的形式對照，因為沒有已發表工作在此 regime
   跑過 vanilla DeepONet，只能自己重跑才是公平比較（chapter04:39 不採他人未經本協定重跑的數字）。

== 顏色規則（三色，各一個意思，不可再增）==
橘 = loss 對著什麼擬合（本頁論點）· 深藍 = 模型主體（結構標示，非好壞）。
（PI-CON 那列已依教授指示移除:literature review 只比較文獻。）
主體用深藍而非橘，是為了不與「loss 擬合對象」搶同一個語意通道。

底部交棒：exactly three 的揭曉在此頁，slide 5 不再提前宣告、slide 7 不再重述。
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
[How the sensor stream enters · 1.5min] 2026-07-18 依指導教授 meeting 新增
（「literature review 一頁一張表，將 literature 分類」）。

**本表的設計意圖**：不是文獻清單，是為 Objective 供證。三張 literature 表各打一個痛點：
  表 1「What prior methods are trained against」→ 打 O1 上半（sensor + PDE only，無全場監督）
  表 2「本頁」                                  → 打 O1 下半（讀 sensor + 任意查詢）與 Gap 2/3
  Gap 4（count/placement/noise）                → 證據放 Motivation 的「Four gaps」頁，本段不重複

**分類軸是機制，不是論文家族** —— 因為痛點是能力缺口，不是誰做了什麼。

口述：「四個欄位就是這個問題需要的四種能力。由上往下讀，每一列都缺至少一欄：
PINN 有 PDE、可任意查詢，但感測器只透過 loss 被評分，網路本身讀不到量測；
operator network 反過來要求輸入取樣在規則網格上；讀得到感測器的網路又回到固定解碼器、
且不帶 PDE；而能處理不規則時鐘的連續時間單元，根本不輸出空間場。
沒有任何一列四欄全有 —— 這就是本研究要填的位置。」

逐格依據（chapter01.tex:92-103 原文）：
- PINN：「the coordinate-MLP backbone takes only (x,t) and sees the sensors solely through a
  loss term, never reading the measurement stream as an input (Gap 2)」
- Operator nets：「the branch expects its input function sampled on a fixed dense grid
  (the FNO spectral mixing requires a regular mesh) rather than ~10² scattered points」；
  PINO「evaluates its residual on a grid」→ 故 PDE 欄標 "on a grid" 而非 ✓/✗
- Sensor-input nets：SHRED「regresses the full state ... through a shallow decoder」、
  Senseiver「cross-attention」；「every one of these is supervised by a full reference field
  ... and imposes no PDE」→ PDE 欄 ✗、readout 欄標 decoder-bound
- Continuous-time cells：CfC「tolerate irregular sampling at single-step cost」但
  「have served control and time-series tasks rather than PDE-constrained spatial
  reconstruction (Gap 3)」→ 無空間場輸出

⚠️ 依教授指示本段不放 PI-CON（literature review 只比較文獻）。四欄全有的那一列在
Objective 段的「Same regime」頁才出現。

⚠️ 期刊名為 references.bib 所載；Chen 2018 (Neural ODE) 為 NeurIPS 會議論文非期刊，
標會議名。被問來源照 bib 回答。
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

<div class="mt-4 px-4 py-3 rounded text-base leading-snug" style="background:rgba(127,16,132,.06); border-left:4px solid #7F1084;">
<b style="color:#7F1084;">Why this matters:</b> the reconstruction is explicitly conditioned on what the instruments measured, while remaining queryable at arbitrary space–time coordinates.
</div>

<FooterLogos />

<!--
[Why operator formulation · 2min] 口述只問一個問題：「量測到底從哪裡進入模型？」
左邊 plain PINN 的 sensor values 只出現在 training loss；推論時 Nθ 只讀 coordinate，
所以量測資訊被烘進 θ。右邊把 sensor history s(·) 與 query (x,y,t) 同時送進 Gθ，
每個輸出都明確 conditioned on measurements。這是 function-conditioned reconstruction 的
概念，不預設 DeepONet、FNO 或 branch/trunk；下一頁才介紹具體 operator family。

⚠️ 不可口述成「新 flow 不需重訓」。chapter01 明確限定 fitting offline and per-case，
cross-realisation amortisation 未驗證。本頁主張的是 measurement-as-input + query-anywhere。
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
[Two operator families · 1min] 2026-07-18 依指導教授 meeting 新增。
教授原話：「介紹 operator 的部分，以及 operator 分成兩大類：FNO 跟 DeepONet，
這樣教授才看得懂什麼是 DeepONet。」

口述順序：
1. 先講**什麼是 operator**（頁首那句）：一般網路學「點 → 值」，operator 學「函數 → 函數」。
   訓練一次就能服務新的輸入函數，不必每個 case 重訓 —— 這是與 plain PINN 的根本差別
   （上一頁講過，這裡只需一句帶過）。
2. **FNO**：把卷積搬到 Fourier 空間、只保留低波數模態，因此一層就有全域感受野。
   代價是輸入必須是**規則網格**上的取樣。
3. **DeepONet**：把映射拆成 branch（讀輸入函數）與 trunk（讀查詢座標），兩者輸出的
   basis 做內積得到該點的值。輸入取樣位置不必規則，查詢座標連續。
4. **落點**（底部橘線）：我們的輸入是 100 個散佈的探針、不是網格；輸出要能在任意座標查詢。
   這兩點正好對上 branch–trunk 的形狀，所以本研究建在 DeepONet 上。

⚠️ 與前後頁的分工：
- 前一頁 (Operator vs. plain PINN) 講「為何要 operator 而非 plain PINN」。
- 本頁講「operator 有哪兩種，我們選哪個、為什麼」。
- 不要在本頁提 CfC / cross-attention / AL —— 那是 Methodology 段 PI-CON 的三項修改，
  提前講會讓委員以為那些是 DeepONet 原本就有的。

⚠️ 圖為架構示意，非實際層數/維度。FNO 實際有 lifting/projection 與多層 Fourier layer，
本頁只畫單層核心以對比 branch–trunk；委員若追問 FNO 細節照此說明。

依據：chapter01.tex:37-40（Table 1.1 operator learning 列：DeepONet [Lu 2021]、FNO [Li 2021]，
原文「a branch sub-network reads the input function, a trunk evaluates the output at any query
coordinate」）與 chapter01:93（DeepONet/FNO「are demonstrated as dense-input forward operators:
the branch expects its input function sampled on a fixed dense grid ... rather than ~10² scattered
points」—— 這是我們為何仍需改造 DeepONet 的伏筆，但**留待 Methodology**）。
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
[Problem formulation · after literature and gap synthesis] This page is deliberately placed after the literature review: the audience now knows why classical, supervised, and existing sensor-input approaches do not meet the deployment regime. The next page quantifies the resolution implied by K = 100; it is not used as an opening claim before the motivation exists.
-->

---

<NavBar active="motivation" />

<SectionTag>§ Motivation · resolution limit under a sparse sensor budget</SectionTag>

# Sensor-count scale and spectral conditioning at K = 100

<style>
.rs { display: grid; grid-template-columns: max-content 1fr; column-gap: 12px; row-gap: 3px;
      align-items: baseline; font-size: 0.77rem; margin-top: 4px; }
.rs .k { color: #6B7280; white-space: nowrap; }
.eb { width: 100%; border-collapse: collapse; font-size: 0.79rem; margin-top: 4px; }
.eb th { text-align: right; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.05em;
         color: #9CA3AF; font-weight: 700; padding: 0 6px 4px 6px; border-bottom: 1px solid #E5E0EC; }
.eb th.l { text-align: left; }
.eb td { padding: 3px 6px; text-align: right; font-variant-numeric: tabular-nums; color: #374151; }
.eb td.l { text-align: left; color: #1F1B2E; font-weight: 600; }
.eb tr.hi td { background: #F7EDF8; color: #7F1084; font-weight: 700; }
</style>

<div class="resolution-old grid grid-cols-5 gap-5 mt-2 items-start">

<div class="col-span-2 space-y-1">

<Card>
<LabelTiny>Sensor Nyquist scale</LabelTiny>
<div class="mt-1 text-xs leading-snug" style="color:#374151;">
The disk |k| ≤ k<sub>max</sub> holds ≈ <b>πk<sub>max</sub>²</b> modes; setting that equal to <b>K</b> gives
</div>
<div class="mt-1 text-center">
<span class="eq" style="font-size: 0.92rem; padding: 0.25rem 0.7rem;">k<sub>max</sub> ≈ √(K/π)</span>
</div>
<div class="rs">
<span class="k">At K = 100</span><span><b style="color:#7F1084;">k<sub>max</sub> ≈ 5.64</b> — a scale, not a wall</span>
<span class="k">Beyond it</span><span>conditioning worsens, κ: 7 → 7×10² <span style="color:#9CA3AF;">(observable to k ≈ 8)</span></span>
</div>
</Card>

<Card>
<LabelTiny>Energy below that scale</LabelTiny>
<table class="eb">
<thead><tr><th class="l">Sensors</th><th>k<sub>max</sub></th><th>DNS energy inside</th></tr></thead>
<tbody>
<tr class="hi"><td class="l">K = 100</td><td>5.64</td><td>98.9 %</td></tr>
<tr><td class="l">K = 200</td><td>7.98</td><td>99.7 %</td></tr>
<tr><td class="l">K = 400</td><td>11.28</td><td>99.9 %</td></tr>
</tbody>
</table>
</Card>

</div>

<div class="col-span-3">
<img :src="'/images/nyquist_recoverability.png'" class="rounded-lg border" style="border-color:#E5E0EC; width: 100%; max-height: 236px; object-fit: contain;" />
<div class="foot mt-1">DNS energy spectrum (a), cumulative fraction (b); dashed line = k<sub>max</sub> = √(K/π). The fractions report energy <b>available</b> below the scale at t = 5, not a proof that higher-k energy is unrecoverable.</div>
</div>

</div>

<style>
.resolution-old { display:none; }
.resolution { display:grid; grid-template-columns:34% 66%; gap:20px; align-items:stretch; margin-top:13px; }
.resolution .cardx { background:rgba(255,255,255,.78); border:1px solid #E5E0EC; border-radius:10px; padding:14px; }
.resolution .tiny { font-size:.72rem; letter-spacing:.07em; text-transform:uppercase; font-weight:700; color:#6B7280; }
.resolution .formula { font-size:1.18rem; color:#7F1084; font-weight:700; text-align:center; margin:16px 0 13px; padding:10px 6px; background:#FAF2FB; border-radius:7px; }
.resolution .copy { font-size:.85rem; color:#374151; line-height:1.4; }
.resolution .evidence { display:grid; grid-template-columns:78px 1fr; column-gap:10px; row-gap:9px; align-items:baseline; margin-top:16px; padding-top:12px; border-top:1px solid #E5E0EC; }
.resolution .evidence b { color:#7F1084; font-size:.82rem; }
.resolution .evidence span { color:#374151; font-size:.80rem; line-height:1.3; }
.resolution .plot img { width:100%; height:270px; object-fit:contain; display:block; }
</style>

<div class="resolution">
  <div class="cardx">
    <div class="tiny">Mode-counting estimate</div>
    <div class="formula">k<sub>sensor</sub> ≡ √(K/π) = 5.64</div>
    <div class="copy">Equating K point measurements to the approximate number πk² of Fourier modes defines a <b>sensor-count scale</b>. It is not a recovery bound.</div>
    <div class="evidence">
      <b>Energy</b><span>98.9% of DNS kinetic energy lies below k<sub>sensor</sub>.</span>
      <b>k ≤ 5</b><span>well-conditioned measurement map, κ ≈ 7.</span>
      <b>k ≤ 8</b><span>still observable but ill-conditioned, κ ≈ 7×10².</span>
    </div>
  </div>
  <div class="cardx plot">
    <div class="tiny">DNS spectrum and cumulative energy at t = 5</div>
    <img :src="'/images/nyquist_recoverability.png'" />
    <div class="copy" style="font-size:.77rem;">Dashed line: k<sub>sensor</sub> = √(K/π). Energy below the line is <b>available</b>, not automatically recoverable.</div>
  </div>
</div>

<FooterLogos />

<!--
[Sensor budget · 2min] 口述收尾（底部 banner 已刪）：「解法是加 sensor，不是加大網路 —— 限制來自資訊，不是架構。」
[Sensor budget · 2min] 兩個視角量化 K=100 觀測能力：①linear system — y = Cu rank-deficient, 650× underdetermined ②CS bound — M ≥ O(s log(N/s)), s≈328 (db4 wavelet), full recovery 需 ~5000 sensors, K=100 差 50×。Implication 精準化：full-field recovery 結構上不可能；productive scope 是 low-band sub-recovery (Nyquist k_max ≈ 5.64) + physics prior 在 null-space 上 regularise。後續 Results 用 sensor Nyquist 與 K-scaling 量化此 scope。
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
<td class="who">Mo &amp; Magri 2025 <span>PC-DualConvNet</span></td>
<td>34</td>
<td>230</td>
<td>✓</td>
<td>128² fixed mesh</td>
</tr>
<tr>
<td class="who">Kelshaw et al. 2022 <span>VDSR CNN</span></td>
<td>34</td>
<td>100</td>
<td>✓</td>
<td>150² fixed mesh</td>
</tr>
<tr>
<td class="who">Parfenyev et al. 2024 <span>coordinate-MLP PINN</span></td>
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
[Literature review 2/2 · 2min] 口述開場：
「同 regime（sensor + PDE、無 full reference field）survey 只找到這三篇，PI-CON 與它們並列。」
—— "the survey finds no others" 是回應委員「怎麼知道這是全部」的關鍵，務必口頭講出。

== 預期提問：「FLRONet 在上一頁，為什麼不在這張表？」（2026-07-18 補）==
**因為納入標準是監督訊號，不是架構。** 本表的 regime 定義（已寫進標題）＝ sensors + PDE
residual、無 full reference field。FLRONet 訓練對著 **paired CFD fields**，屬全場監督，
故歸在 slide 6（該頁的軸正是「對什麼擬合」）。論文對應：chapter01:101 是共享 sensor+PDE
監督的三篇；chapter01:103 另立一段講「query-anywhere 且讀 sensor、卻仍需 dense full-field
supervision」的 Senseiver 與 FLRONet。

⚠️ 這題有後半段，要主動接上：FLRONet 是 chapter01:103 明載的
「**the closest published architecture to the present branch--trunk readout**」——
架構上最接近的一篇。委員會追問「那你跟它比了嗎？」
答：**架構家族相同，但無法做同 regime 的數值對打** —— 它需要成對 CFD 場，在本研究的
工程約束（現場無 DNS）下根本訓練不起來。差別不在架構優劣，在監督訊號能不能在現場取得。
真正的 branch–trunk 對照是內部的 B0 vanilla DeepONet（主結果頁 8.23% → 5.71%，
−2.53 pp, p=3.0×10⁻⁷），因為沒有已發表工作在此 regime 跑過 vanilla DeepONet，
只能自己重跑才是公平比較（chapter04:39 不採未經本協定重跑的他人數字）。

== 口述三個 take-away（2026-07-16 從頁面移除的三張卡，改用講的）==
① Reynolds number：最接近的一篇仍低 7.7×（1.3×10³ vs 10⁴），兩篇 CNN 低 300×（34 vs 10⁴）。
② Measurement model：Mo & Magri 用 2.3× 於我們的 probe 數（230 vs 100）；
   Parfenyev 沒有固定測站，但**量測量遠多於我們** —— 3×10⁴ 個隨機 (r, t) 樣本，
   平均每 snapshot 150 點（vs 我們 100 個固定測站），那是任何 rig 都裝不出來的
   量測模型（chapter01:101「not one a rig can install」）。
   ⚠️ 講法要小心：不可說「Parfenyev 不需要 sensor」——它用得比我們多，優勢在於
   我們的量測模型工程可實現、它的不可實現。這是 trade-off 不是我們贏在資訊量。
③ 本頁的落點：**沒有任何 surveyed work 同時做到 query-anywhere + sensors-as-input + Re=10⁴**
   —— 表格右兩欄加上 Re 欄一起看就是這個結論，指著表講即可，不需要再印一次。

== 表格設計（2026-07-16 重做）==
原本 7 欄且同一組橘/紫被用在四種不同意思上（Re 的橘、probes 的橘、readout 的橘、
residual 的橘都代表「比我們差」）—— 每格都是重點就等於沒有重點。現在：
  - 欄位砍到 5 欄：Work / Re / Probes / Sensors as input / Readout
  - 移除 Architecture 獨立欄（併為 Work 下的灰色小字）與 NS residual 欄
    （NS residual 與 Readout 高度相關：query-anywhere 必然是 autodiff，fixed mesh 必然是
     FD/pseudospectral，多一欄不增資訊；且「都有 PDE residual」正是本頁 same-regime 的前提，
     已寫在標題下那句）
  - 顏色只留一個意思：**紫色 = PI-CON 那一列**。其餘全中性。

逐篇出處（2026-07-15 查證）：
- Mo & Magri 2025 (arXiv 2409.00260): Re=34、80 input + 150 general sensors (≈0.9%)、
  128² grid、PC-DualConvNet (U-Net + Fourier branch)、residual 用 2nd/4th-order FD。
  原文報 relative ℓ₂ 5.51 ± 0.34 %（非 KE MAPE）。
- Kelshaw 2022 (arXiv 2210.17319): ν=1/34、10×10=100 觀測、150² grid、VDSR + bicubic、
  可微 pseudospectral residual。
- Parfenyev 2024 (arXiv 2404.01193): Re≈1.3×10³、Ndata=3×10⁴ (≈0.2%)、coordinate MLP
  7×250、autodiff residual、scattered 量測。
  ✅ 2026-07-18 直接查證 arXiv 全文，原文：「measurements reveal the velocity field at
  N_data points, which are **randomly scattered within the observation area and time
  interval**」「around 0.2% of the total data... **On average, this corresponds to 150
  points per snapshot** compared to 65536 points at full resolution」。
  → 確認 (a) 隨機散布非固定測站、(b) 每 snapshot 150 點。
  ⚠️ 本頁 Probes 欄一度誤寫「none」——那是錯的（它有 3×10⁴ 個量測），且對我們不利
  （讓 Parfenyev 看起來不需要量測 = 比我們強）。已改為「150 / snapshot」。
窮盡性：chapter01:99 界定同 regime 者恰為三篇，此表即全集。

⚠️ 兩個已知問題（尚未修 thesis）：
1. 舊版本頁曾寫「Mo & Magri KE MAPE ~23% → ours 5.71%」——該 23% 全 repo 無來源，
   原文亦無近似值（唯一「over 20%」是其 loss 變體間的相對比較）。已移除。
   chapter04:39 本就聲明不採用未經本協定重跑的他人數字為證據。
2. ~~chapter01:99 稱三篇「each returns a fixed mesh」對 Parfenyev 為誤述~~
   → **已修正，2026-07-18 複查**：chapter01:101 現行文字已正確區分
   「The two convolutional works return a fixed mesh」與「The PINN evaluates anywhere,
   but its 3×10⁴ measurements are (r,t) pairs scattered...」。論文無需再改。
不要在口試宣稱與 Mo & Magri 的 head-to-head 數值優勢：指標不同、Re 差 300 倍。
-->

---

<NavBar active="objective" />

<SectionTag>§ Objective</SectionTag>

# Research objective

<div class="mt-2 text-base leading-snug" style="color:#374151;">
Reconstruct 2-D turbulent flow from sparse (u, v) sensors and the Navier–Stokes residual, with <b style="color:#7F1084;">no DNS field</b> in training, then map how <b style="color:#7F1084;">count, placement, noise</b> govern quality.
</div>

<style>
.ob { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 12px; }
.ob .goal { font-size: 0.85rem; line-height: 1.4; color: #374151; margin-top: 6px; }
.ob .crit { margin-top: 8px; padding-top: 7px; border-top: 1px solid #EFEAF2; }
.ob .crit .lb { font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.05em;
                color: #9CA3AF; font-weight: 700; }
.ob .crit ol { margin: 4px 0 0 0; padding-left: 15px; font-size: 0.78rem; line-height: 1.42; color: #374151; }
.ob .crit li { margin-bottom: 2px; }
</style>

<div class="ob">

<Card>
<LabelTiny style="color:#7F1084;">(O1)&nbsp; Accurate &amp; fast reconstructor</LabelTiny>
<div class="goal">Engineering-grade from <b>sensor + PDE</b> only, queryable at any (x, t).</div>
<div class="crit">
<div class="lb">Criterion</div>
<ol>
<li>KE rel-err <b>&lt; 10 %</b>, n = 5</li>
<li><b>−2 pp</b> vs vanilla DeepONet, <b>p &lt; 0.01</b></li>
<li>Reconstruction <b>≥ 5×</b> faster than re-solving</li>
</ol>
</div>
</Card>

<Card>
<LabelTiny style="color:#7F1084;">(O2)&nbsp; Count sets the resolution</LabelTiny>
<div class="goal">Recoverable band set by <b>sensor count</b>, not architecture.</div>
<div class="crit">
<div class="lb">Criterion</div>
<ol>
<li>Effective cutoffs <b>bracket √(K/π)</b> ≈ 5.64 at K = 100</li>
<li>K ∈ {100, 200, 400} shifts the band along <b>√(K/π)</b></li>
</ol>
</div>
</Card>

<Card>
<LabelTiny style="color:#7F1084;">(O3)&nbsp; Placement &amp; noise set reliability</LabelTiny>
<div class="goal">Placement and noise change reliability, <b>not feasibility</b>.</div>
<div class="crit">
<div class="lb">Criterion</div>
<ol>
<li>All three placements within target, placement σ <b>≥ 3×</b> seed σ</li>
<li>Noise to <b>10 %</b> of sensor σ stays within target</li>
</ol>
</div>
</Card>

</div>

<FooterLogos />

<!--
[Objective · 1.5min] 對齊論文三軸 arc：工具(PI-CON) + sensing-configuration 系統研究（數量/位置/噪音）。
上方一句話：用 PI-CON 從稀疏 sensor + NS residual 重建流場（無 DNS 全場），並系統研究 sensing config 如何決定品質。
三個 Objective（先 qualitative goal、後 falsifiable criterion）：
  O1 重建器（準＋快）：sensor+PDE only 達 engineering grade，任意點單次前傳。criterion KE<10% n=5 / dominant lever ≥2pp @p<0.01 / 單次前傳 ≥5× 快於 forward-solving。
  O2 數量軸：可重建波數由 sensor 數量決定，非架構。criterion k_max^sensor=√(K/π)≈5.64 @K=100；K=100/200/400 cutoff 隨 √(K/π) 移動。
  O3 位置&噪音軸：placement/noise 影響 reliability 不影響 feasibility。criterion 三 placement 皆 engineering-grade、σ_placement≥3×σ_training；noise 到 10% 仍 engineering-grade。
⚠️ 2026-07-16 移除底部的 Contribution 區塊 —— 三重重複：
(a) 「surveyed 中唯一結合 query-anywhere + sensor-only-with-physics @ Re=10⁴」已在
    slide 7（Same-regime works）的「No surveyed work combines all three」講過；
(b) 「PI-CON = CfC branch + cross-attn + AL-continuity」是 slide 12（Three additions
    to DeepONet）整頁的內容；
(c) 貢獻條列在 §Conclusion 有專頁。
本頁專責「要達成什麼 + 怎麼判定失敗」，架構與貢獻留給後面。

口述橋接（頁面不印）：「這三個目標分別由架構、數量軸、位置與噪音軸回答 —— 先看架構。」
論文 §Objective 結尾有 \paragraph{Contribution}（thesis/CLAUDE.md 要求），那是論文體例；
投影片有專頁，不需在此重複。
-->

---

<NavBar active="method" />

<SectionTag>§ Application case · the Kolmogorov flow</SectionTag>

# Reference units are prescribed; flow scales are realised

<style>
.ndflow { display:grid; grid-template-columns:43% 57%; gap:22px; margin-top:12px; align-items:start; }
.ndflow .flow-gif { width:100%; height:330px; object-fit:contain; border:1px solid #E5E0EC; border-radius:10px; background:rgba(255,255,255,.76); }
.ndflow .caption { margin-top:5px; text-align:center; color:#6B7280; font-size:.74rem; }
.ndflow .box { border:1px solid #E5E0EC; border-radius:9px; background:rgba(255,255,255,.80); padding:10px 13px; margin-bottom:8px; }
.ndflow .label { font-size:.70rem; letter-spacing:.07em; text-transform:uppercase; color:#6B7280; font-weight:700; margin-bottom:6px; }
.ndflow .refs { display:flex; flex-wrap:wrap; gap:6px 18px; color:#374151; font-size:.80rem; line-height:1.35; }
.ndflow .refs b { color:#0F2D52; }
.ndflow .vars { display:grid; grid-template-columns:1fr 1fr; gap:5px 14px; color:#1F1B2E; font-size:.82rem; }
.ndflow .vars span { white-space:nowrap; }
.ndflow .ns { border-color:#C9A6CC; background:#FAF2FB; }
.ndflow .equation { text-align:center; color:#0F2D52; font-size:.81rem; line-height:1.55; white-space:nowrap; }
.ndflow .eqstack { display:flex; flex-direction:column; gap:2px; margin-top:2px; }
.ndflow .eqstack .primary { color:#0F2D52; font-size:.76rem; font-weight:600; }
.ndflow .eqstack .secondary { color:#6B7280; font-size:.72rem; }
.ndflow .eqstack .params { white-space:nowrap; }
.ndflow .case { margin-top:4px; text-align:center; color:#6B7280; font-size:.74rem; }
</style>

<div class="ndflow">
  <div>
    <img :src="'/images/kolmogorov_dns_vorticity_anim.gif'" class="flow-gif" />
    <div class="caption">DNS vorticity · periodic unit square</div>
  </div>
  <div>
    <div class="box">
      <div class="label">Reference normalisation <span style="font-weight:400;text-transform:none;letter-spacing:0;">(chosen units)</span></div>
      <div class="vars">
        <span>x<sup>*</sup> = x / L<sub>0</sub></span>
        <span>t<sup>*</sup> = tU<sub>0</sub> / L<sub>0</sub></span>
        <span><b>u</b><sup>*</sup> = <b>u</b> / U<sub>0</sub></span>
        <span>p<sup>*</sup> = p / U<sub>0</sub>²</span>
      </div>
      <div class="case">L<sub>0</sub> = box edge and U<sub>0</sub> = control velocity define the code units; Re<sub>0</sub> ≡ U<sub>0</sub>L<sub>0</sub>/ν = 10⁴.</div>
    </div>
    <div class="box">
      <div class="label">Flow-derived characteristic scales <span style="font-weight:400;text-transform:none;letter-spacing:0;">(measured from DNS)</span></div>
      <div class="vars">
        <span>λ<sub>f</sub> = L<sub>0</sub>/k<sub>f</sub> = <b>0.5 L<sub>0</sub></b></span>
        <span>U<sub>rms</sub> = <b>0.503 U<sub>0</sub></b></span>
        <span>t<sub>f</sub> = λ<sub>f</sub>/U<sub>rms</sub> = <b>0.995 T<sub>0</sub></b></span>
        <span>Re<sub>f</sub> = U<sub>rms</sub>λ<sub>f</sub>/ν = <b>2.51×10³</b></span>
      </div>
    </div>
    <div class="box ns">
      <div class="label" style="color:#7F1084;">Nondimensional system actually solved <span style="font-weight:400;text-transform:none;letter-spacing:0;">(stars omitted hereafter)</span></div>
      <div class="eqstack">
        <div class="primary">∂<b>u</b>/∂t + (<b>u</b>·∇)<b>u</b> = −∇p + (1/Re<sub>0</sub>)∇²<b>u</b> + <b>f</b></div>
        <div class="primary">∇·<b>u</b> = 0</div>
        <div class="secondary params">Ω = [0,1]² &nbsp; · &nbsp; ν = 10⁻⁴ &nbsp; · &nbsp; <b>f</b> = (0.1 sin 4πy, 0)</div>
      </div>
    </div>
  </div>
</div>

<FooterLogos />

<!--
[Kolmogorov flow (DNS solution) · 1.5min] 2026-07-18 依指導教授 meeting 改版。

本頁區分 reference units 與 flow-derived scales。L0（box edge）與 U0（control velocity）是
無因次化 convention；λf 與 Urms 則分別由 forcing wavelength 與實際 DNS 場決定。
遠端實際 trajectory（20% burn-in 後）計得 Urms=0.502625、tf=0.994776、Ref=2513.13。

口述順序（照卡片由上而下）：
「上面 L0、U0 是用來定義 code units 的 reference scales，所以取 1 本身沒有問題；
但它們不是流場自然產生的特徵。這個 Kolmogorov flow 真正由物理解出的尺度是 forcing
wavelength λf=0.5L0 與 realised Urms=0.503U0，因此 injection-scale Reynolds number
是 2.51×10³，與 prescribed control Re0=10⁴ 分開報。」

壓力 p 是 kinematic pressure，因此以 Uref² 無因次化，不另寫密度 ρ。

⚠️ 本頁是全簡報「無因次 convention」的宣告點：其後各頁的 t = 5、KE、rel-L₂ 等一律不帶單位。

⚠️ Ref 比 control Re0 小，因為 λf=0.5L0 且 realised Urms=0.503U0；兩者不可混稱。
→ 若委員進一步問「那這個場到底多湍流」：誠實答能譜斜率 −4.61、[0,1]² 箱內無延伸慣性range
   （chapter03:113），本研究定位在工程重建而非湍流物理。
-->

---

<NavBar active="method" />

<SectionTag>§ Application case · numerical setup</SectionTag>

# How the reference data are generated

<style>
.st { display: grid; grid-template-columns: max-content 1fr; column-gap: 12px; row-gap: 2px;
      align-items: baseline; font-size: 0.75rem; margin-top: 3px; }
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
<tr><td class="c">Grid convergence</td><td class="m">ΔKE ≤ 0.064 %</td><td class="j"><span class="ok">verified</span></td></tr>
<tr><td class="c">Time-step convergence</td><td class="m">Δu<sub>rel-L₂</sub> = 7.0×10⁻⁶</td><td class="j"><span class="ok">verified</span></td></tr>
<tr><td class="c">Spectral divergence</td><td class="m">‖∇·u‖<sub>∞</sub> ≤ 7×10⁻¹⁵</td><td class="j"><span class="ok">round-off</span></td></tr>
</tbody>
</table>
<div class="cap2"><b style="color:#0F2D52;">DNS verified:</b> independent grid, time-step, and constraint checks passed.</div>
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
[Setup · 1.5min] 教授要求補 CFD 必要參數。本頁只講「數值設定」。
⚠️ Governing equations / forcing / Re 定義那張卡已於 2026-07-16 移除 —— 改由前一頁
（"Kolmogorov flow at Re = 10⁴"）專責，兩頁曾重複。不要再把方程加回本頁。
- DNS algorithm：pseudo-spectral with 2/3 dealiasing [Orszag 1971; Boyd 2001] + ETDRK4 fp64 [Cox–Matthews 2002; Kassam–Trefethen 2005]
- Grid 256²、Δt = 2.5e-4、snapshot Δt_s = 0.025、N_t = 201、T = 5
- DNS verification 在頁面只保留三個直接對照，避免把 criterion 當成 convergence 證明：
  1. Grid：獨立 N=256 與 N=1024、相同 IC 的 post-spin-up KE 最大相對差 0.064%；
     K≤5.64 band 的 spectrum difference 0.05%。來源 docs/grid_independence_re10000.md、
     docs/gi_test_re10000_analysis.md。
  2. Time step：N=256、T=0.5，將 Δt=2.5e-4 減半至 1.25e-4；t=0.5 的
     rel-L2(u)=7.01e-6、rel-L2(v)=6.78e-6，時間離散誤差比空間誤差小約 160×。
  3. Divergence：1024² spectral solver 每個存檔點以 spectral derivative 計算
     max|∇·u|；home-gpu /home/latteine/gi_test_re10000/N1024.log 的最大值為
     6.88e-15（t=0），其後約 3e-16–1.5e-15。頁面保守寫 ≤7e-15。

預備追問：k_max·η 如何得到？20% burn-in 後 ε=ν〈ω²〉=6.27e-3，
η=(ν³/ε)^(1/4)=3.55e-3；stored N=256 且 2/3 dealias，
k_max,phys=(N/3)(2π/L)=536.2，因此 k_maxη=1.91（run N=1024 為 7.61）。
這是 resolution sanity check，不把 Pope-2000 criterion 單獨當成 2-D grid-convergence proof；
主證據是上面的 direct N-ref comparison。

預備追問：原本的 0.13 是什麼？它是 U_rms Δt/Δx_run =
0.503×2.5e-4/(1/1024)=0.129，應稱 CFL_rms，不是 CFL_max，因此已從頁面移除。

⚠️ T=5 只有約 2.51 box-scale turnover times；本研究把它當單一 transient reconstruction
trajectory，不主張 statistically stationary ensemble。五個 training seeds 只量 optimisation
stochasticity，不能補償 flow-statistical window。
- Sparse-sensor card：K = 100, QR-pivot POD [Manohar 2018], operator target G_θ, loss 只用 sensor + NS（不偷 DNS / ω / E(k)）
右 col 維持 sensor placement 圖。把 engineering target（KE/div/k_f amp 數字）移到 §Results，§Setup 不背具體閾值。
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
<span class="k">Purpose</span><span class="v"><b style="color:#7F1084;">large-scale proxy for QR-pivot only</b></span>
<span class="k">Equation</span><span class="v">filtered NS with SGS stress and linear friction</span>
<span class="k">Solver</span><span class="v">pseudo-spectral, 2/3 dealiasing, RK2 Heun, fp64</span>
<span class="k">Grid / horizon</span><span class="v"><b>N = 256</b>, T<sub>end</sub> = 50</span>
<span class="k">Closure</span><span class="v">Bardina scale-similarity + spectral hyperviscosity</span>
<span class="k">Friction</span><span class="v"><b style="color:#E97132;">r = 2.86×10⁻²</b> <span style="color:#9CA3AF;">— absent from DNS</span></span>
<span class="k">Cost</span><span class="v">approximately <b>1/16 of DNS</b></span>
</div>
</Card>

<Card>
<LabelTiny>What was verified</LabelTiny>
<table class="lesvf">
<tbody>
<tr><td class="c">Incompressibility</td><td class="m">‖∇·u‖<sub>max</sub> = 2.29×10⁻¹³</td><td class="j"><span class="ok">verified</span></td></tr>
<tr><td class="c">Alias control</td><td class="m">tail decay = 5.14×10³²</td><td class="j"><span class="ok">verified</span></td></tr>
<tr><td class="c">Statistical window</td><td class="m">T<sub>end</sub>/<span class="raw">τ</span><sub>int</sub> = 4.9 &lt; 10</td><td class="j"><span class="warn">not established</span></td></tr>
</tbody>
</table>
<div class="lescap"><b style="color:#0F2D52;">Accepted for placement only:</b> no claim of statistically converged LES data.</div>
</Card>

</div>

<div>
<Card style="padding-top:.6rem; padding-bottom:.6rem;">
<img :src="'/images/les_T50_vorticity_with_sensors.png'" style="width:100%; max-height:250px; object-fit:contain;" />
<div class="lespipe"><b>LES large-scale field</b> <span style="color:#C9C6D0;">→</span> QR-pivot <span style="color:#C9C6D0;">→</span> <b style="color:#7F1084;">K = 100 fixed locations</b></div>
<div class="lescap">DNS supplies (u, v) only at these coordinates for the offline study.</div>
</Card>
</div>

</div>

<FooterLogos />

<!--
[LES placement · 1.5min] This page follows the DNS setup because it explains where the K=100 locations come from.
DNS remains the offline reference; LES is a cheaper deployment-time proxy used only by QR-pivot. The LES is
numerically resolved and divergence-free for its discretisation, but the statistical window is not established
(T_end/tau_int=4.9<10). This does not justify LES statistics; downstream placement quality is evaluated against DNS.
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

  <rect x="570" y="279" width="220" height="46" rx="7" fill="#FAF2FB" stroke="#C9A6CC" stroke-width="1.3"/>
  <text x="680" y="297" text-anchor="middle" fill="#7F1084" style="font-size:12.5px;font-weight:700;">OPTIMIZER</text>
  <text x="680" y="315" text-anchor="middle" fill="#374151" style="font-size:11px;">SOAP + Schedule-Free</text>

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
[Architecture · 1.5min] 2026-07-18 依指導教授 meeting 改版：
(a) 標題不再用 "Three additions"；(b) 加上底色圖例（深藍＝沿用的 DeepONet backbone、
橘＝本研究新增、虛線藍框＝DeepONet 內積前的 basis tensors）；(c) 三張說明卡移到下一頁；
(d) **PI-CON 這個名稱在本頁首次正式登場**（教授指定的位置，前面各頁一律不提前用）。

口述：「先看顏色：深藍是 DeepONet 原有的結構 —— MLP trunk、兩組 basis，以及最後的
inner product；橘色是加入的 Fourier embedding、CfC branch 與 distance-biased
cross-attention。資料有兩條路徑：100 個感測器的時間訊號由 CfC 編碼，query coordinate
先做 Fourier embedding 再進 trunk；cross-attention 讓 query 取回 sensor context。
虛線框不是新 layer，而是兩條路徑產生、交給原始 DeepONet inner product 的 basis tensors。
輸出在訓練時進入 sensor MSE + NS residual，SOAP optimizer 再以 back-propagation 更新全部 θ；
這條紫色回授線是 training loop，inference 時不執行。」

顏色語意核對：
- Fourier embedding：非原始 DeepONet，橘色；vanilla B0 為公平比較也保留相同 encoding，
  不代表它是 DeepONet 原生元件。
- CfC branch、cross-attention + distance bias：PI-CON 新增，橘色。
- MLP trunk、basis-factorisation role、inner-product readout：DeepONet 原有，深藍／虛線藍。
- Input/output tensors：淡紫中性色，不屬於 inherited/added module。

⚠️ 本頁只講「是什麼」，三個新增元件「為何需要」在下一頁。不要在此展開，否則兩頁重複。
-->

---

<NavBar active="method" />

<SectionTag>§ Methodology · why each addition is needed</SectionTag>

# What each addition is for

<style>
.ad { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 16px; }
.ad .n { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: #D97757; }
.ad .t { font-size: 1.0rem; font-weight: 700; color: #1F1B2E; margin-top: 3px; }
.ad .p { font-size: 0.86rem; color: #374151; line-height: 1.45; margin-top: 7px; }
.ad .g { font-size: 0.78rem; color: #9CA3AF; margin-top: 7px; padding-top: 6px; border-top: 1px solid #EFEAF2; }
</style>

<div class="ad">

<div>
<div class="n">Addition 1</div>
<div class="t">CfC branch</div>
<div class="p">Reads each sensor's <b>time history</b>, and stays differentiable when the clock is uneven, so the record enters as an input rather than a loss term.</div>
<div class="g">Closes the sensor-input and uneven-clock gaps</div>
</div>

<div>
<div class="n">Addition 2</div>
<div class="t">Cross-attention</div>
<div class="p">Maps <b>sparse sensors to any query point</b> using a distance bias, so a query draws on the sensors near it instead of a fixed grid.</div>
<div class="g">Closes the sparse-to-dense gap</div>
</div>

<div>
<div class="n">Addition 3</div>
<div class="t">Augmented Lagrangian</div>
<div class="p">Applies an <b>adaptive penalty on ∇·u</b>, tightening incompressibility as training proceeds rather than fixing its weight in advance.</div>
<div class="g">Keeps the field physical without a reference</div>
</div>

</div>

<div class="mt-6 px-4 py-3 rounded" style="background: rgba(127,16,132,0.06); border-left: 4px solid #7F1084; font-size:0.92rem; line-height:1.5; color:#374151;">
At K = 100, a vanilla DeepONet does reconstruct, but not well enough to be useful: <b>8.23 %</b> KE error against the <b>10 %</b> engineering target. The three additions bring it to <b style="color:#7F1084;">5.71 %</b> — a <b>2.52 percentage-point</b> gap at <b>p = 3×10⁻⁷</b>.
</div>

<FooterLogos />

<!--
[Why each addition · 1.5min] 2026-07-18 新增（教授要求把三張卡從架構圖頁移出）。

⚠️ 底部結論的措辭經過校準,**不可寫成「vanilla DeepONet 做不出來 / 訓不起來」**：
2×2 ablation 實測 B0 vanilla DeepONet = 8.23 %,它是可訓練的,只是達不到 10 % 工程門檻。
教授原話「做不出來」經 2026-07-18 確認指的是「**重建結果不好**」,故頁面寫
「does reconstruct, but not well enough to be useful」。若寫成無法訓練,委員翻到
主結果頁的 ablation 表就能反駁。
數字來源:主結果頁 B0 8.23 % → B3 5.71 %,−2.53 pp,p = 3.0×10⁻⁷（chapter04, n=5 seeds）。

三個 Gap 的對應（chapter01:108-118）：CfC → Gap 2/3、cross-attention → Gap 2、AL → Gap 1。
-->



<!--
[Architecture · 2min] 教授九點 (7) (9) 落實：把 Architecture 寫成「接 Objective、補 vanilla DeepONet 缺口」三件套 narrative，不在這頁堆 AI mathematical 細節。
Gap → Addition → Why-it-hits-(Ox) 三欄表：
  Gap (a) DeepONet branch 吃固定 grid snapshot → CfC continuous-time branch → 對齊 (O1) sensor-only feasibility
  Gap (b) inner product 沒 spatial prior → cross-attention with ‖r‖ bias → 對齊 (O2) streaming inference
  Gap (c) PDE residual soft penalty → Augmented Lagrangian on continuity → 對齊 (O3) physical consistency
底部一行交代 trunk Fourier embed [Tancik 2020] + GradNorm [Chen 2018] + 3.14M params。
CfC 內部公式、cross-attn 內部公式留 backup slides，這張只講 narrative。
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

<div class="grid grid-cols-2 gap-5 mt-2">

<Card>
<LabelTiny>① LIQUID NEURAL NETWORK (LNN) [Hasani 2021]</LabelTiny>

<div class="mt-1 text-xs leading-snug">
h relaxes toward a target A, the <b>decay rate depends on the input</b> — a "liquid" time constant:
</div>

<div class="mt-1" style="font-size: 0.95em;">

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

<div class="mt-1" style="font-size: 0.95em;">

$$h(t + \Delta t) = \sigma \odot f_1 + (1 - \sigma) \odot f_2$$

</div>

<div class="mt-1" style="font-size: 0.95em;">

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
[CfC introduction · backup 1min] 口述開場（標題下小字已刪）：「vanilla DeepONet 的 branch 吃固定 grid snapshot，
我們的 sensor 是不等間隔取樣的 time series，所以把 branch 換成 closed-form continuous-time RNN。」 教授九點 (9) — CFD lab 不重 AI 細節：把 CfC 介紹改成「補 vanilla DeepONet 的 time-signal 缺口」narrative。
頂部一句話：vanilla DeepONet branch 吃固定 grid snapshot，我們的 sensor 是 irregular time series，所以換成 closed-form continuous-time RNN。
卡 1：LNN ODE 形式 + 為何 vanilla ODE solver 在 autograd 內貴。
卡 2：CfC analytical closed-form + O(1) per step + autograd 安全。
底部「為何要 CfC」三條 chip 移除（已在頂部 narrative + 卡 2 末尾標明）。
-->

---

<NavBar active="method" />

<SectionTag>§ Method · cross-attention readout (closing the sparse-to-dense gap)</SectionTag>

# Cross-attention — closing the "sparse-to-dense" gap

<div class="grid gap-5 mt-2" style="grid-template-columns: 1.12fr 0.88fr;">

<Card>
<LabelTiny>① CROSS-ATTENTION READOUT [VASWANI 2017]</LabelTiny>

<div class="mt-1 text-xs leading-snug" style="color:#374151;">
For each target point <b style="color:#0F2D52;">q = (x, t)</b>, attention asks: <b>which sensors are most informative here?</b>
</div>

<div class="mt-2" style="font-size: 0.98em;">

$$A_{qk}=\operatorname{softmax}_k\!\left(\mathbf Q_q^{\mathsf T}\mathbf K_k/\sqrt{d}+b(r_{qk})\right)$$

</div>

<div class="mt-1 text-xs" style="display:grid; grid-template-columns:max-content 1fr; column-gap:9px; row-gap:2px; align-items:baseline; color:#374151;">
<b style="color:#0F2D52;">Q<sub>q</sub><sup>T</sup>K<sub>k</sub>/√d</b><span>learned query–sensor similarity</span>
<b style="color:#D97757;">b(r<sub>qk</sub>)</b><span>periodic spatial-distance bias</span>
</div>

<div class="mt-1" style="font-size: 0.97em;">

$$\mathbf c(q)=\sum_{k=1}^{K} A_{qk}\,\underbrace{\mathbf V_k}_{\text{sensor information}}$$

</div>

<div class="mt-1 px-2 py-1 rounded text-xs" style="background:#FAF2FB; color:#374151; text-align:center;">
<b style="color:#7F1084;">Compare</b> q with all sensors <span style="color:#C9C6D0;">→</span>
<b style="color:#7F1084;">normalise</b> scores into A<sub>qk</sub> <span style="color:#C9C6D0;">→</span>
<b style="color:#7F1084;">combine</b> V<sub>k</sub> into c(q)
</div>

</Card>

<Card>
<LabelTiny>② TWO FLUID-SPECIFIC MODIFICATIONS</LabelTiny>

<div class="mt-1 text-xs leading-snug"><b>Causal lookup</b> → <b style="color:#0F2D52;">streaming-deployable</b></div>

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

<style>
/* 兩條 attention 公式相鄰，KaTeX 自帶的 display margin 疊起來會頂出頁尾。
   只收這一頁的公式間距，不動別頁（Slidev 會把此 style scope 到本 slide）。 */
.slidev-page-13 .katex-display { margin: 0.15em 0 !important; }
</style>

<FooterLogos />

<!--
[Cross-attention introduction · backup 1min] 口述開場（標題下小字已刪）：「inner product 是 global、沒有 spatial prior；
Q 來自 trunk，K 與 V 來自 sensor token —— 所以是 cross 不是 self。」（chapter02:290-292） 少談 Transformer 內部細節，照 thesis §2.3 的骨架講：
「Two modifications adapt it to fluid reconstruction: an isotropic relative-position bias and a causal lookup over sensor time.」(chapter02:257)
頂部一句話：vanilla DeepONet inner product 是 global、沒 spatial prior，所以換成 cross-attention 做 sparse-to-dense readout（chapter02:135 原話）。
卡 1：保留 cross-attention 的原始數學定義，不改成另一個概念；只把兩個 score 項直接標成
      query–sensor similarity 與 spatial-distance bias，並用 Compare → Weight → Combine 三步白話解讀。
      為何叫 cross 不叫 self：self-attention 的 Q/K/V 同源，這裡 Q 來自 trunk（查詢點）、
      K 與 V 來自同一個 sensor token H_q[k] 走 W_K / W_V 兩個投影（chapter02:290-292）。
      卡上用顏色編碼：Q 深藍（trunk）、K 與 V 橘（sensor token，同源）。
      √d_hidden 是 softmax 溫度縮放，不是 V；用 \big/ 寫成一行避免根號被 \frac 分母縮小而誤讀成 V。
      b_qk = MLP_relpos(LayerNorm(r_qk))；
      r_qk 先 fold 回環面再取 smooth norm，ε=10⁻⁸ 是為了避免 query 落在 sensor 上時 second-order autograd 出 NaN（chapter02:279）。
卡 2：兩項 fluid-specific 修改，改用圖不用公式（searchsorted/clamp 是實作細節，同心圓是幾何，兩者用文字講都不直觀）。
      被問實作時再答 chapter02:263 的 n*(q) = clamp(searchsorted({t_n}, t_q, right=True) − 1, 0, N_t − 1)。
      (a) causal lookup：binary search 只讀 t ≤ t_q → streaming-deployable；
      (b) 為何 isotropic 而非 directional：forcing 讓流場統計非等向，但 directional bias 會把 QR-pivot sensor 的
      非均勻 x 分布學成假的方向性注意力（chapter02:268）。這是被問「為何不用 (r_x, r_y)」時的標準答案。
移除原本「Self-attention vs cross-attention」對比（純 AI 細節，CFD lab 不感興趣）。
移除原本「卡 2：CFD analogue — learnable RBF interpolant」：該類比全論文不存在（chapter02 提 RBF 0 次），
且 RBF 在論文裡只是被打敗的 baseline（chapter04:219，pointwise u rel-L₂ 低 47–74%）— 講它等於自找交叉詰問。
-->

<!--
[LES generation · 2min] 教授九點 (4) (5) 落實：LES 也要把 CFD 重要參數寫清楚 + 解析度/穩定度判斷標準。
左卡 1 — filtered NS + Domain/BC（雙週期，與 DNS 一致）+ SGS closure（Bardina scale-similarity [Bardina 1980; Sagaut 2006] + spectral hyperviscosity）+ Solver（pseudo-spectral + 2/3 dealiasing, RK2 Heun fp64；DNS 才用 ETDRK4）+ N=256, T_end=50, cost ≈ 1/16 DNS
左卡 2 — LES 品質 gate。2026-07-16 用 scripts/check_les_quality.py 對
data/les/kolmogorov_les_Re10000_N256_T50_standalone.npy 實測：
  [1] incompressibility  max‖∇·u‖ = 2.29e-13 < 1e-10（solver fp64 診斷值，非從 float32 場重算）PASS
  [2] no aliasing pile-up  譜末端衰減比 5.14e32 > 1e6  PASS
  [3] statistical window  ⚠️ **腳本報 τ_int = 4.28 → T_end/τ_int = 11.68 PASS，此判定不可採信**

⚠️⚠️ 2026-07-17 更正（頁面已改；先前投影片寫「statistical convergence ✓ · 11.7 ≥ 10」是錯的）：
scripts/check_les_quality.py 的 τ_int 是**從那支 50 s 紀錄自己**估的 → 系統性低估。
若真實 τ_int ≈ 10 s，50 s 只有約 5 個相關時間，自相關會提早過零，積分必然偏小 ——
**太短的紀錄無法自己揭露它太短**，是自我實現的假通過。
論文 chapter03:193 用正確做法：另跑一支 **T_end = 400 s** 診斷（N=128 較便宜，同 closure /
friction / dealiasing）量得 **τ_int = 10.1 s** → 50/10.1 = **4.9 < 10，未達**；N_eff ≈ 2.5，
且該 400 s 診斷到 t=400 仍有微弱能量上飄 → 「no horizon reachable at this cost reaches a
statistically steady state」。tab:les_verification 亦已標為「Statistical window (not met; see text)」。
→ 口試不可宣稱 LES 統計收斂。正確說法：**LES 只需提供大尺度空間結構供佈點，不需統計收斂**
（頁面 Role 那行已改成這個口徑）。
→ TODO: scripts/check_les_quality.py 的 [3] 應改為要求外部提供 τ_int（或拒絕在
   T_end/τ_int < 10 時給 PASS），否則它會繼續對每支短 LES 發假通過。
⚠ 以下三個舊 gate 已被證偽，禁止再講（CLAUDE.md LES_Quality_Anti_Patterns）：
  ✗「T/t_eddy ≥ 5、EXP-221 達 26.5」—— LES 帶 linear friction −r·u，KE 由 forcing–friction 平衡主導，
     時間尺度是 1/(2r) = 17.5，比 eddy time (~2–3.5) 長一個數量級。用錯時間尺度。
  ✗「KE plateau / rel_change(KE) < 5%」—— 回看窗由 save_interval 決定，結構上不可能失敗。
  ✗「spectral overlap within 2× DNS on k ∈ [2, N/3]」—— LES 有 friction、DNS 沒有，能量平衡不同；
     實測 k ∈ [2,85] 全帶 0/84 個波數落在 [0.5,2] 內，物理上不可能通過。
被問「統計窗夠不夠」：**誠實答未達**（T_end/τ_int = 4.9 < 10），並說明為何不影響本研究 ——
LES 的角色是提供佈點所需的大尺度空間結構，不是提供收斂統計量；佈點品質的證據是下游結果
（EXP-245 LES placement KE 5.71 ± 0.12%，與 DNS-oracle 4.68% 同級、皆遠低於 10% 門檻），
不是 LES 自身的統計收斂。不要答 eddy-turnover（用錯時間尺度）。
底部 Pill 用 final fair-comparison 口徑：LES placement 是 EXP-245 main pipeline，KE 5.71 ± 0.12%；不要再用舊 placement-ablation 的 12.36% / 9.40% 作主張。
-->

---

<NavBar active="method" />

<SectionTag>§ Training · closing the physics-consistency gap</SectionTag>

# Augmented Lagrangian on ∇·u

<div class="grid grid-cols-2 gap-5 mt-3 text-sm">

<Card>
<LabelTiny>AUGMENTED LAGRANGIAN (AL) ON CONTINUITY</LabelTiny>

<div class="mt-2" style="font-size: 0.82em;">

$$\mathcal{L}_{\text{AL}} \;=\; \lambda\,C \;+\; \tfrac{\rho}{2}\,C^2, \qquad C \,\equiv\, \mathcal{L}_{\text{cont}}$$

</div>

<div class="mt-1" style="font-size: 0.82em;">

$$C \,=\, \mathbb{E}_{\text{collocation}}\big[(\partial_x u + \partial_y v)^2\big] \,\ge\, 0$$

</div>

<div class="mt-1" style="font-size: 0.82em;">

$$\lambda \,\leftarrow\, \mathrm{clip}\big(\lambda + \rho\,\bar{C},\; 0,\; \Lambda_{\max}\big)$$

</div>

<div class="mt-2 text-xs" style="color:#6B7280;">
ρ = 0.1, Λ<sub>max</sub> = 10, C̄ = EMA of C (β = 0.5), updated every 100 steps.<br>
<b style="color:#E97132;">C ≥ 0 ⇒ λ rises monotonically</b> — an accumulated-multiplier schedule, not a textbook equality-constraint Lagrangian (whose λ would change sign).
</div>
</Card>

<Card>
<LabelTiny>CFD ANALOGUE &amp; OBSERVED EFFECT</LabelTiny>

<div class="mt-2 text-xs" style="display:grid; grid-template-columns:max-content 1fr; column-gap:12px; row-gap:5px; align-items:baseline;">
<b style="color:#7F1084;">SIMPLE / PISO</b><span>pressure-correction Poisson, <b>exact, pointwise</b></span>
<b style="color:#7F1084;">Our AL (λ)</b><span>ascent on the mean residual, <b>in expectation</b></span>
</div>

<div class="mt-1 text-xs" style="color:#6B7280;">
an analog, not an algorithmic equivalent
</div>

<div class="mt-3 pt-2" style="border-top: 1px solid #E5E0EC;">
<LabelTiny>Is the constraint doing anything?</LabelTiny>
<div class="mt-1" style="display:grid; grid-template-columns:1fr max-content; column-gap:12px; row-gap:4px; align-items:baseline; font-size: 0.83rem; font-variant-numeric:tabular-nums;">
<span style="color:#6B7280;">DNS, full cascade</span><span style="color:#9CA3AF;">1.04 %</span>
<span style="color:#6B7280;">DNS, band-limited to k ≤ 16</span><span style="color:#9CA3AF;">0.38 %</span>
<span style="color:#1F1B2E; font-weight:600;">PI-CON, same bandwidth</span><span style="color:#7F1084; font-weight:700;">0.39 %</span>
</div>
<div class="mt-2 text-xs leading-snug" style="color:#6B7280;">
At the FD floor of its resolved bandwidth — <b>not</b> below DNS.
</div>
</div>

</Card>

</div>

<FooterLogos />

<!--
[Continuity AL · 1.5min] 左卡 AL formulation 完整：penalty C 是 continuity 平方期望、dual ascent
λ ← clip(λ + ρC̄, 0, Λ_max)、ρ=0.1 Λ_max=10、C̄ 為 C 的 EMA(β=0.5) 每 100 步更新。

⚠️ 2026-07-17 修正兩處錯誤（實測佐證，見下）：
(1) 原式寫 `L_AL = L + λC + (ρ/2)C²` —— 把總損失塞進 AL 項自己的定義裡，代入 eq:total_loss
    會遞迴。正解 `L_AL = λC + (ρ/2)C²`（論文 eq:al_loss / chapter02:390；程式碼
    src/pi_con/losses.py:111 `return self.lambda_ * C + 0.5 * self.rho * C ** 2`）。
(2) 原本寫「λ grows when continuity is violated, decays once C is small」—— **錯**。
    C 是平方的平均恆 ≥ 0，losses.py:127 的 `(lambda_ + rho*ema_C).clamp(0, clip)`
    因此單調不減，λ **永遠不會 decay**。實測 12 個 run、共 10 萬+ 步，**0 次下降**。
    λ 是「趨緩」不是「下降」：C → 0 使增量消失（EXP-245 seed42：λ 在 5k/10k/15k/20k 步
    為 0.3383 / 0.3649 / 0.3776 / 0.3857，增量 0.027 → 0.013 → 0.008）。

預期提問「Λ_max = 10 為什麼？λ 會不會撞到上界？」
→ 誠實答：**從來沒有撞到**。實測 λ 收斂在 0.386，只有 clip 的 3.86%（EXP-245 seed42,
   artifacts/lab/exp245_b3_seed42/metrics.jsonl, 20k 步）。跨 12 個 run 的 λ_max 落在
   clip 的 2.8–65%，主線那批（B3/B0/B1/B2/4-head）全在 3.7–4.5%；最高的 exp250-pinn-tanh
   到 65% 仍未 binding。所以 Λ_max 是個未 binding 的 safety guard，不是作用中的機制；
   λ 停下來是因為殘差降到近零，不是因為被夾住。
   ⚠️ 論文 chapter02:400 原本誤寫成「until it saturates the clip」，已於 2026-07-17
   同步改為「with Λ_max bounding it from above; the multiplier settles as the residual
   falls rather than by reaching that bound」。若引用舊版 PDF 需注意此差異。

口述（頁面已移除，被問 ρ 才講）：「把 ρ 拉到 1 可以把 divergence 再壓到 0.28%，
但要付出場精度的代價 —— 這個旋鈕是活的，我們選 0.1 是取平衡。」
右下三行數字就是「constraint 有沒有在作用」的答案，不需要再用文字複述一次。右卡 CFD analogue — SIMPLE/PISO 的 pressure correction p' 是 Lagrange multiplier，我們的 λ 用 gradient ascent 取代 Poisson 解；enforce in expectation 而非 exactly on grid。觀測 effect 用 final protocol 說法：EXP-245 divergence ratio 0.39 ± 0.006%，接近 resolved-bandwidth finite-difference floor。
-->

---

<NavBar active="method" />

<SectionTag>§ Training · optimisation &amp; multi-task balancing</SectionTag>

# SOAP + Schedule-Free + GradNorm — why second-order matters

<div class="grid grid-cols-2 gap-5 mt-3 text-sm">

<Card>
<LabelTiny>SOAP + SCHEDULE-FREE &nbsp;<span class="opacity-50">[Wang 2025, Defazio 2024]</span></LabelTiny>

<div class="mt-3" style="display:grid; grid-template-columns:max-content 1fr; column-gap:12px; row-gap:8px; align-items:baseline;">
<b style="color:#7F1084;">SOAP</b><span>Shampoo-style <b>2nd-order preconditioner</b>, Adam in the preconditioner eigenbasis</span>
<b style="color:#7F1084;">Schedule-Free</b><span>Polyak–Ruppert averaging, no lr decay</span>
<b style="color:#7F1084;">Why both</b><span>anisotropic valleys at Re = 10⁴, Adam zigzags, SOAP overshoots, SF averaging stabilises</span>
</div>

<div class="mt-3 px-3 py-2 rounded text-xs leading-snug" style="background: rgba(127,16,132,0.06);">
<b style="color:#7F1084;">Better stability than vanilla Adam</b> in the multi-task PINN setting.
</div>
</Card>

<Card>
<LabelTiny>GRADNORM &nbsp;— auto-weighting 4 loss tasks &nbsp;<span class="opacity-50">[Chen 2018]</span></LabelTiny>

<div class="mt-3 text-base">

$$\|w_i\,\nabla\!_{\theta_r}\,\mathcal{L}_i\| \;\propto\; (\mathcal{L}_i / \mathcal{L}_i^{(0)})^{\alpha},$$

</div>

<div class="mt-3" style="display:grid; grid-template-columns:max-content 1fr; column-gap:12px; row-gap:8px; align-items:baseline;">
<b style="color:#7F1084;">Every 1 000 steps</b><span>equalise <b>gradient-norm magnitude</b> across {data, NS-u, NS-v, cont}</span>
<b style="color:#7F1084;">Why</b><span>hand-tuned weights are brittle — too small ⇒ data overfit, too large ⇒ near-zero collapse</span>
</div>
</Card>

</div>

<FooterLogos />

<!--
[Optimisation · 2min] 左卡 SOAP+SF — Shampoo 2nd-order + Polyak averaging。chaotic NS valleys 需要 2nd-order；SOAP+SF 比 vanilla Adam 在 multi-task PINN 更穩定（thesis §Optimization）。（-20% KE 屬 EXP-030 log、thesis 未收，故 slide 只寫質性。）右卡 GradNorm — 4-task gradient-norm equalisation，每 1000 步調權重，避免 hand-tuned brittle。Init 物理 0.01 (1% of data weight)，會自己 ramp up。下一張講 AL 怎麼補 continuity。
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
<tr><td>Fourier embed</td><td>16 harmonics</td><td>N<sub>q</sub> × 128</td></tr>
<tr><td>MLP trunk</td><td>1 residual block</td><td>N<sub>q</sub> × 256</td></tr>
<tr><td>Cross-attention</td><td>1 layer, 1 head</td><td>N<sub>q</sub> × 100 × 256</td></tr>
<tr><td>Inner product</td><td>3 fields × rank 256</td><td>N<sub>q</sub> × 3</td></tr>
</tbody>
</table>
<div class="dimsum">
<span>Total trainable parameters <b>3.14 M</b></span>
<span>Query grid <b>128 × 128</b> <span class="cite">(DNS: 256 × 256)</span></span>
</div>
</Card>

<Card>
<LabelTiny>Training</LabelTiny>
<div class="pgrid">
<div class="k">Supervision</div><div class="v"><b>sensor MSE + NS residual only</b></div>
<div class="k">Optimiser</div><div class="v">SOAP + Schedule-Free, lr = 10⁻³, warm-up 2 000</div>
<div class="k">Collocation</div><div class="v">1 024 points per step</div>
<div class="k">Budget</div><div class="v">20 000 iterations × <b>n = 5 seeds</b> (42, 1, 2, 3, 4)</div>
<div class="k">Hardware</div><div class="v"><b style="color:#7F1084;">Single</b> RTX 3090 (24 GB), ~2 h 45 m per seed</div>
</div>
</Card>

</div>


<FooterLogos />

<!--
[Model & training config · 1min] §Method 最後的 reproducibility summary。

⚠️ 2026-07-16 重新設計：原本四張卡，其中兩張半是重複的，已移除 ——
  - Flow & DNS 卡（domain / Re / forcing / DNS solver / T）→ 已在 slide 10（Kolmogorov
    flow case：Ω=[0,1]²、A=0.1、k_f=2、Re=UL/ν=10⁴）與 slide 11（setup at a glance：
    pseudo-spectral + ETDRK4 fp64、run 1024² → stored 256²、Δt_s=0.025、N_t=201）講過。
  - Sensors 卡（K=100 (u,v)、LES-derived QR-pivot）→ 已在 slide 11 與 slide 15（LES proxy）講過。
本頁只保留「別處沒有的數值」：model 尺寸與 training 預算。前面各頁講「為什麼」，本頁給「多少」。

頁面所有數值已於 2026-07-18 直接對照 configs/stable/exp_245.toml 與 src/pi_con/{encoders,decoder}.py：
spatial encoder = 1 residual block, 256 channels, internal hidden 512；temporal branch = 1 CfC layer ×
256 neurons；sensor-token mixing = 2 self-attention blocks × 4 heads；query trunk = 1 residual block ×
256 neurons；decoder cross-attention = 1 head；operator rank = 256；d_emb=128（16 harmonics）；
3.14M params · evaluation query grid 128²（DNS reference 256²）· lr=10⁻³ warm-up 2000 · 1024 collocation ·
20 000 iters × n=5 (seeds 42,1,2,3,4) · single RTX 3090 24GB ~2h45m/seed。

⚠️ 2026-07-19 改寫左卡：原本是 component × depth × hidden width × additional setting 的四欄表，
列出 512-neuron inner layer / operator rank 256 / 16 harmonics 這些前面從未出現過的量，
粒度比架構頁細一階。現在改成「Module × Size × Tensor」，**每一列都對應架構圖（slide 16）
的一個方塊**，第三欄就是圖上已經印過的 tensor shape → 講到這頁時可以說「這就是剛才那張圖，
補上尺寸」，不再有憑空冒出的數字。
模組歸屬已核對 src/pi_con/{operator,encoders}.py 與 exp_245.toml：
num_spatial_cfc_layers=1（SpatialSetEncoder，1 residual block）；
num_temporal_cfc_layers=1 + num_token_attention_layers=2 × token_attention_heads=4
（兩個 self-attention block 屬 TemporalCfCEncoder，**不是** spatial encoder — 舊 slide 的
「Sensor mixing」獨立成列會讓人以為它在 encoder 那側）。
未印在表上、被問再口述：spatial encoder 內層 hidden = 2×d_model = 512。

未印在本頁但 backup 有（被問再翻）：SOAP β=(0.9, 0.999)、precond_freq=2、Polyak averaging；
GradNorm 每 1000 步更新、EMA 0.9、init 權重 (1, 0.01, 0.01, 0.01)；AL ρ=0.1、λ_clip=10。
AL 與 SOAP/GradNorm 的「為什麼」在 slide 16 / 17，本頁不重述。
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
<div class="sym">KE MAPE</div><div class="def"><b>headline</b>; KE(t) = ½∫<sub>Ω</sub>(u²+v²) dx</div>
<div class="sym">RMSE<sub>u,v</sub>(t)</div><div class="def">time-resolved, absolute (not normalised)</div>
<div class="sym">div ratio</div><div class="def">‖∇·u‖₂ / ‖∇u‖<sub>F</sub><sup>DNS</sup></div>
</div>

<div class="foot mt-1">Derivatives: 4th-order central differences, 128² grid.</div>
</Card>

<Card>
<LabelTiny>TRAINING LOSS &nbsp;<span class="opacity-60">(GradNorm-balanced [Chen 2018])</span></LabelTiny>

<div class="eqbox">

$$\mathcal{L}(\theta) = w_d \mathcal{L}_{\text{data}} + w_{\text{NS},u} \mathcal{L}_{\text{NS},u} + w_{\text{NS},v} \mathcal{L}_{\text{NS},v} + w_c \mathcal{L}_{\text{cont}} + \textcolor{#E97132}{\mathcal{L}_{\text{AL}}}$$

</div>

<div class="eqbox" style="border-left-color:#E97132;">

$$\mathcal{L}_{\text{AL}} = \lambda\,C + \tfrac{\rho}{2}\,C^2, \qquad C \equiv \mathcal{L}_{\text{cont}}$$

</div>

<div class="ngrid">
<div class="sym">ℒ<sub>data</sub></div><div class="def">MSE on the K = 100 sensor channels</div>
<div class="sym">ℒ<sub>NS,u</sub> , ℒ<sub>NS,v</sub></div><div class="def">NS momentum residual at collocation points</div>
<div class="sym">ℒ<sub>cont</sub></div><div class="def">∇·u residual — the same C the AL acts on</div>
<div class="sym" style="color:#E97132;">ℒ<sub>AL</sub></div><div class="def">adaptive continuity pressure via λ, <b>outside</b> GradNorm</div>
<div class="sym">w<sub>d</sub> , w<sub>NS</sub> , w<sub>c</sub></div><div class="def">GradNorm-balanced weights</div>
</div>

<div class="mt-3 pt-2 text-xs leading-snug" style="border-top: 1px solid #E5E0EC; color:#374151;">
<b style="color:#7F1084;">Invariant</b>,  DNS field never enters ℒ.
</div>
</Card>

</div>

<FooterLogos />

<!--
[Evaluation metrics · 1.5min] 左卡只保留後續結果頁實際引用的四個量：
  (1) 全時空 global rel-L₂(u,v,ω)：主結果表與各 sensing ablation 使用。
  (2) KE MAPE：headline bulk metric，所有主比較使用。
  (3) 逐時 RMSE_u,v(t)：Temporal diagnostics 的速度誤差曲線使用；用絕對誤差避免分母隨時間改變。
  (4) div ratio：continuity diagnostic，與 resolved-bandwidth FD floor 比較。
rel-L∞ 在簡報結果中完全未使用，故刪除；t*=5 rel-L₂ 只是既有 rel-L₂ 在特定快照的取值，
不再冒充一個獨立 metric。最後標明：4 階中央差分、128² eval grid。
右卡 Loss formulation：4-task GradNorm weighted sum **+ L_AL**。底部紅線：「DNS field 從不入 L」標明工程不可遷移性。
注意：avoid 「approximately / matches」這類 hardness/marketing 語。

⚠️ 2026-07-17 修正：原式漏了 `+ L_AL`，變成純 penalty method 的長相 —— AL 在 §Methodology
講了一整頁，到這裡卻在公式裡看不到，委員會問「AL 到底加在哪」。依論文 eq:total_loss
（chapter02:337）與 eq:al_loss（chapter02:390）補回，並經 src/pi_con/losses.py:111
`return self.lambda_ * C + 0.5 * self.rho * C ** 2` 核實。
講法：「連續性進 loss 兩次 —— 一次是 GradNorm 平衡的 soft term w_c·L_cont，一次是 AL 對
**同一個** C 施加的自適應壓力 L_AL；AL 刻意放在 GradNorm 之外，因為 GradNorm 只平衡
gradient 量級、不對散度設下限（chapter02:389）。」
-->

---

<NavBar active="results" />

<SectionTag>§ Results · main result · architectural value</SectionTag>

# Main result — 2×2 ablation at n = 5

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
<LabelTiny>2×2 ablation, KE MAPE (%, n = 5, lower is better)</LabelTiny>

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

<div class="foot mt-2">Bottom row flips sign: CfC costs <b style="color:#E97132;">+0.99</b> alone, buys <b style="color:#7F1084;">−1.32</b> with cross-attention.</div>
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
<LabelTiny>Welch t-test &nbsp;<span class="opacity-60">(n = 5 seeds)</span></LabelTiny>
<div class="rg">
<div class="k">B3 &nbsp;PI-CON</div><div class="n" style="color:#7F1084;">5.71 ± 0.12 %</div>
<div class="k">B0 &nbsp;vanilla DeepONet</div><div class="n">8.23 ± 0.22 %</div>
<div class="k tot">gap</div><div class="n tot" style="color:#7F1084;">−30.7 % rel</div>
</div>
</Card>

</div>

</div>

<FooterLogos />

<!--
[Architectural ablation · 2min] 長條圖：4 個架構變體 B0/B1/B2/B3 的 KE MAPE 比較（按 KE 排序）。右上 KE decomposition：cross-attn −1.21pp（dominant lever）、CfC +0.99pp（worse alone）、interaction −2.31pp、sum −2.53pp。右下 multi-seed n=5 t-test：B3 vs B0 −2.53pp（−30.7% relative）、t=22.9、p=3.0×10⁻⁷、Cohen's d=14.5（統計顯著性從投影片移來，字太小委員看不清，改口述）。v-clicks：①兩個 component 都 essential、cross-attn 強 lever ②operator framework > raw capacity (PINN 3.24M < DeepONet 1.28M)。
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
<div>, Main vortex structure recovered</div>
<div>, Small scales (k &gt; 5) smoothed — sensor Nyquist scale</div>
<div>, Error sits on <b>high-shear edges</b>, not random</div>
<div>, |u, v error| ≪ |ω error| (ω amplifies derivatives)</div>
</div>
</Card>

<div class="mt-3 text-xs leading-snug" style="color:#6B7280;">
Source, EXP-245 baseline (B3 + LES_T50 + 1024 collo), seed 42 field viz, metrics n = 5.
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
[Field reconstruction · 口述（caption 已刪，只留 source）：colourbar 上 DNS 與 PI-CON 共用 ±max，
error panel 獨立縮放 —— 委員問「顏色能不能直接比」時照此答。
[Field reconstruction · 2.5min] 合併版：左 title + key observations bullet (k_f mode recovered, mid/high k smoothed, error on high-shear edges, u/v < ω error)，右上 velocity 6-panel (u/v × DNS/PI-CON/Error) + 右下 vorticity 3-col。Speaker：「展示主結果視覺面 — 先看 velocity 場（u/v）DNS 與 PI-CON 視覺幾乎一致，error magnitude 小；再看 vorticity 場是 derivative quantity，amplifies error 但仍 capture 主結構。Error 集中 high-shear edges 是 sensor Nyquist 上限造成。Velocity 圖高度 = vorticity 兩倍（2 row vs 1 row），讓單 panel 視覺等大。」
-->

---

<NavBar active="results" />

<SectionTag>§ Results · vorticity error interpretation</SectionTag>

# Error structure across wavenumbers

<style>
.bg2 { display: grid; grid-template-columns: max-content 1fr; column-gap: 14px; row-gap: 4px; align-items: baseline; margin-top: 6px; margin-bottom: 0; }
.bg2 .k { font-size: 0.90rem; color: #6B7280; white-space: nowrap; }
.bg2 .v { font-size: 0.90rem; color: #1F1B2E; line-height: 1.3; }
</style>

<div class="grid grid-cols-5 gap-4 mt-3">

<div class="col-span-3">
<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<LabelTiny>Band-resolved relative error vs time &nbsp;<span class="opacity-60">(EXP-245, n = 5)</span></LabelTiny>
<img :src="'/images/band_energy_rel_error_vs_time.png'" class="mt-1" style="width: 100%; max-height: 248px; object-fit: contain;" />
</Card>
</div>

<div class="col-span-2 space-y-2">

<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<LabelTiny>Key metrics &nbsp;<span class="opacity-60">(EXP-245, n = 5)</span></LabelTiny>
<div class="bg2">
<div class="k">KE MAPE</div><div class="v"><b style="color:#7F1084;">5.71 ± 0.12 %</b></div>
<div class="k">u rel-L₂</div><div class="v">13.65 ± 0.06 %</div>
<div class="k">v rel-L₂</div><div class="v">17.52 ± 0.10 %</div>
<div class="k">ω rel-L₂</div><div class="v">41.79 ± 0.12 %</div>
<div class="k">div ratio</div><div class="v"><b style="color:#7F1084;">0.39 ± 0.006 %</b></div>
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
[Vorticity error interpretation · 2min] 口述接回第 8 頁：「k ≤ 5 這條線就是第 8 頁的 sensor Nyquist
k_max ≈ 5.64；越過它 conditioning 急遽變差（κ 7 → 7×10²），加大網路補不回來。」注意別說成
「modes 比 measurements 多 / 不可觀測」—— 那要到 k ≈ 8 才成立（appendix06 的 SVD：2K=200 個
(u,v) 觀測、M=196，k ≲ 8 內每個 mode 都 full-rank 可觀測）。原本這裡有張 Ceiling 卡寫同樣的
5.64 與同樣的結論，與第 8 頁逐字重複、且右欄已擠爆，故移除改為口述。
左 metrics 用 EXP-245 main (LES_T50, 20k, n=5)：KE 5.71 ± 0.12%, ω rel-L₂ 41.79%, div ratio 0.39%。右三個 Card 解讀：①DNS reference 有什麼 (k_f forcing + cascade) ②PI-CON 抓到什麼 (主 vortex + k_f mode 對的振幅相位，小尺度 smoothed) ③Error 結構性 (集中在 high-shear edges, 不是 random noise)。後面 spectral analysis 量化這個 information bound。
-->

---

<NavBar active="results" />

<SectionTag>§ Results · EXP-245 baseline (B3 + LES_T50, 1024 collo)</SectionTag>

# Temporal diagnostics

<div class="grid grid-cols-2 gap-4 mt-3">

<Card>
<img :src="'/images/kinetic_energy_vs_time.png'" class="rounded" style="max-height: 252px; width: 100%; object-fit: contain;" />
<div class="foot mt-1">KE MAPE <b style="color:#7F1084;">5.71 ± 0.12 %</b> (n = 5), follows DNS decay 0.161 → 0.122, IC warm-up t &lt; 2 s.</div>
</Card>

<Card>
<img :src="'/images/uv_rmse_vs_time.png'" class="rounded" style="max-height: 252px; width: 100%; object-fit: contain;" />
<div class="foot mt-1">u, v RMSE <b style="color:#7F1084;">0.115 → 0.03</b> (n = 5, ±1σ), absolute, no denominator, flat after t ≈ 3 s.</div>
</Card>

</div>

<FooterLogos />

<!--
[Temporal diagnostics · 1.5min] 兩張圖：KE(t)（MAPE 5.71 ± 0.12%, n=5, 追 DNS chaotic decay
0.161→0.122 m²/s²）、velocity RMSE u/v(t)（0.115→0.03 m/s, ±1σ band n=5）。
div ratio 0.39% 接近 resolved-bandwidth FD floor。

⚠️ 2026-07-17 改動：右圖由 rel-L₂ 換成 **RMSE**（絕對值, m/s）。理由（plot_multiseed_envelope.py
plot_combined docstring）：rel-L₂ 的分母是 DNS 分量量值，而該量值不是定常的 —— t=3→5 之間 DNS
在分量間重分配能量（KE_u +46.5%, KE_v −60.3%，總 KE 只動 +3.3%），‖v‖_rms 掉 37%。
於是 v 的 rel-L₂ 在窗尾上翹（實測 t=3 的 11.08% → t=5 的 16.38%），但絕對誤差其實是平的
（v RMSE 0.035 → 0.032 m/s）。舊圖那條上翹曲線讀起來像「重建在衰敗」，是分母縮小的假象。
被問「為什麼不用 rel-L₂／為什麼跟表上的 13.65% / 17.52% 對不起來」→ 答：表報的是整個時空場的
global rel-L₂，此圖報逐時絕對 RMSE，兩者是不同的量；論文 fig:main_trajectories 的第 4 格
（DNS velocity-component magnitude）就是為了讓表上的 rel-L₂ 能從圖反推回來而存在。
被問「那個能量重分配是什麼造成的」→ 誠實答：未確立。forcing f = (A sin(k_f y), 0) 只作用在 u 上，
但 forcing-mode 振幅與 KE_v 的相關性很弱（r = −0.26），不足以歸因。

圖檔來源：scripts/plot_multiseed_envelope.py 的 `uv_rmse_vs_time.png` spec，資料為
artifacts/exp245_seeds/eval_245_seed{a..e}_final（與論文 fig:main_trajectories 同一批資料）。

⚠️ 2026-07-17 資料修正：本頁與論文圖原先用 `eval_245_seed{a..e}_mac`，那批是拿
`checkpoints/picon_kolmogorov_step_20000.pt` 評估的 —— 該檔的 `schedulefree_mode='train'`，
存的是 ScheduleFree 的 train-mode 權重（給 resume 用），evaluator 會 WARN「inference quality
比 final.pt 差 5-30%」。log 與主表的數字則來自 lab 用 `picon_kolmogorov_final.pt`（eval-mode）
跑的 eval，兩者 83/97 個 tensor 不同（最大相對差 10%）→ 圖與表本來不同源。
現已全部改用 final.pt 重跑：per-seed KE / u / v / ω 與 log 表四位小數精確吻合
（KE 5.9035 / 5.6751 / 5.6491 / 5.7144 / 5.5882），div ratio 也從錯誤的 0.50% 回到 0.39%，
與本頁講的數字一致。RMSE 與 rel-L₂ 對此差異不敏感（v rel-L₂ 11.10→11.08%），故上述物理論述不變。

⚠️ 2026-07-16 改動：原本三張圖並排（KE / uv / E(k)），每張只有 1/3 寬，但 PNG 內的
label 與 legend 字級是照全寬設計的，縮到 1/3 後委員看不清。改為兩張並排（max-height
180 → 270px），每張面積約 2 倍。

移除的 E(k) at t=5 並未損失論證：K-scaling 頁（下一張）的三連能譜已含 K=100 的
E(k) 與 Nyquist 截止線，且那張是專門為投影片畫的（字級較大），本來就比這張清楚。
band-resolved 的 k≤5 / mid-high 飽和數字則在「Error structure」頁的 key metrics。
若要恢復三圖，正解是重畫 PNG 加大字級，不是把圖再縮小。

⚠️ 已移除「v > u: forcing acts on u」—— 該歸因全 thesis 查無，且 cross-attention 的
isotropic kernel 是未排除的競爭解釋（見已停用的 velocity-error backup 頁）。
-->

---

<NavBar active="results" />

<SectionTag>§ Results · sensor placement axis (O3)</SectionTag>

# Sensor placement without DNS access

<style>
.pl { width: 100%; border-collapse: collapse; font-size: 0.90rem; margin-top: 12px;
      font-variant-numeric: tabular-nums; }
.pl th { text-align: right; font-weight: 700; color: #7F1084; font-size: 0.90rem;
         padding: 0 12px 7px 12px; border-bottom: 2px solid #7F1084; white-space: nowrap; }
.pl th:first-child { text-align: left; }
.pl td { padding: 10px 12px; border-bottom: 1px solid #E5E0EC; color: #374151; text-align: right; }
.pl td:first-child { text-align: left; color: #1F1B2E; }
.pl tr.main td { background: rgba(127, 16, 132, 0.09); font-weight: 700; color: #7F1084; }
.pl .no  { color: #7F1084; font-weight: 700; }
.pl .yes { color: #E97132; font-weight: 700; }
.pl .sub { font-weight: 400; color: #9CA3AF; }
</style>

<div class="text-xs mt-1" style="color:#6B7280;">
Same B3 backbone, 1024 collocation, 20 k iterations, n = 5, sensor values always come from the K = 100 positions only.
</div>

<table class="pl">
<thead>
<tr>
<th>Placement strategy</th>
<th>DNS full field to place?</th>
<th>KE MAPE (%)</th>
<th>u / v rel-L₂ (%)</th>
</tr>
</thead>
<tbody>
<tr class="main">
<td>LES T = 50 &nbsp;<span class="sub">main pipeline</span></td>
<td class="no">No</td>
<td>5.71 ± 0.12</td>
<td>13.65 / 17.52</td>
</tr>
<tr>
<td>DNS QR-pivot &nbsp;<span class="sub">oracle</span></td>
<td class="yes">Yes</td>
<td><b>4.68 ± 0.06</b></td>
<td>15.34 / 18.10</td>
</tr>
<tr>
<td>Random uniform &nbsp;<span class="sub">no-effort fallback</span></td>
<td class="no">No</td>
<td>7.95 ± 0.68</td>
<td>17.20 / 21.62</td>
</tr>
</tbody>
</table>

<div class="grid grid-cols-3 gap-4 mt-5 text-xs">

<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<LabelTiny>KE and pointwise L₂ trade off</LabelTiny>
<div class="mt-1 leading-snug" style="color:#374151;">
The oracle takes <b>KE</b>. LES takes <b>pointwise L₂</b> — and needs no DNS field to place.
</div>
</Card>

<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<LabelTiny>Placement is worth 2.24 pp</LabelTiny>
<div class="mt-1 leading-snug" style="color:#374151;">
LES over random: <b style="color:#7F1084;">−2.24 pp</b> of KE for one cheap LES run.
</div>
</Card>

<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<LabelTiny>Reliability, not feasibility</LabelTiny>
<div class="mt-1 leading-snug" style="color:#374151;">
Spread from <b>placement</b> (± 0.68) is <b style="color:#E97132;">6×</b> the spread from training seeds (± 0.11). All three clear 10 %.
</div>
</Card>

</div>

<FooterLogos />

<!--
[Placement · 2min] O3 位置軸。表格即證據 —— 論文 §placement (sec:placement_analysis)
本身也是純表格、無圖。

欄位語意取自 chapter04.tex:347 的「DNS full field for placement?」（Yes/No），一欄即可，
不需要 pre-deployment cost + engineering deployable 兩欄互相重複。
數字：chapter04.tex:351 / :386（random u 17.20±1.42, v 21.62±2.07）、tab:main_metrics。

三張卡：
1. trade-off —— oracle 贏 KE、LES 贏 pointwise L₂（chapter04:357），不可寫成 LES 全面勝。
2. −2.24 pp —— LES 相對 random 的 KE 增益（chapter04:45 原文；實算 7.95 − 5.71 = 2.24）。
3. variance —— ± 0.68（跨 5 種佈點）vs ± 0.11（跨 training seed）＝ 6.2×，對應
   chapter04:45「reduces placement-induced variance roughly sixfold」。這是 O3
   「placement 影響 reliability 不影響 feasibility」的量化依據，三者皆 < 10 % 門檻。

⚠️ 已移除原本右欄的 les_T50_vs_dns_spectrum.png：
   (a) 它回答「這個 LES 好不好」，屬 slide 12 的範圍；本頁講 placement 結果，該圖不支撐本頁主張。
   (b) 其 caption「Leading POD modes overlap within 2× on k ∈ [2, N/3]」正是
       thesis/CLAUDE.md 明列的 LES 禁項 ——「不可拿 LES 能譜與 DNS 比對當 gate；
       friction 使兩者能量平衡不同，物理上不可能通過，吻合也不構成 LES 品質證據」。
   同一條 gate 亦仍掛在 thesis chapter03.tex:208 的 tab:les_params，待處理。
-->


---

<NavBar active="results" />

<SectionTag>§ Results · vs an open-loop forward-CFD forecast</SectionTag>

# Trajectory divergence of the forward-CFD reference

<style>
.fc { display: grid; grid-template-columns: max-content 1fr 1fr; column-gap: 14px; row-gap: 5px;
      align-items: baseline; margin-top: 8px; margin-bottom: 0; }
.fc .hd { font-size: 0.78rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.04em; text-align: right; }
.fc .k  { font-size: 0.78rem; color: #6B7280; white-space: nowrap; }
.fc .v  { font-size: 0.78rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.fc .bad { color: #E97132; font-weight: 700; }
.fc .good { color: #7F1084; font-weight: 700; }
</style>

<div class="grid grid-cols-5 gap-4 mt-2">

<div class="col-span-3">
<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<LabelTiny>Velocity rel-L₂ &nbsp;<span class="opacity-60">, full window, no selection</span></LabelTiny>
<img :src="'/images/forward_cfd_divergence.png'" class="mt-1" style="width: 100%; max-height: 240px; object-fit: contain;" />
</Card>
</div>

<div class="col-span-2 space-y-2">

<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<LabelTiny>u rel-L₂ &nbsp;<span class="opacity-60">, start → end</span></LabelTiny>
<div class="fc">
<div></div><div class="hd">t = 0</div><div class="hd">t = 5</div>
<div class="k">Forward-CFD</div><div class="v">5.2 %</div><div class="v bad">152.8 %</div>
<div class="k">PI-CON</div><div class="v">26.9 %</div><div class="v good">7.28 %</div>
</div>
<div class="mt-1 text-xs leading-snug" style="color:#374151;">
Open-loop starts <b>better</b>, <b style="color:#E97132;">diverges 29×</b>.<br/>
Sensor-conditioned <b style="color:#7F1084;">converges</b>.
</div>
</Card>

<Card style="padding-top: 0.6rem; padding-bottom: 0.6rem;">
<LabelTiny>Statistics look fine</LabelTiny>
<div class="mt-1 text-xs leading-snug" style="color:#374151;">
At t = 5: <b>KE −3.85 %</b>, <b>enstrophy +3.46 %</b>, spectrum within <b>≈10 %</b>.
</div>
<div class="mt-1 text-xs leading-snug" style="color:#6B7280;">
On the attractor, <b style="color:#E97132;">wrong phase</b>, σ<sub>u</sub>/σ<sub>v</sub> 2.32 → 0.90.
</div>
</Card>

</div>

</div>

<div class="foot text-[10px]" style="margin-top: 2px;">Gappy-POD init (rank 40, Everson & Sirovich 1995), open-loop, not matched assimilation, basis from <b>200 offline DNS snapshots</b> — more than PI-CON sees.</div>

<FooterLogos />

<!--
[Forward-CFD · 2min] 底部 note 精簡後的完整口徑：這不是自創方法，是兩個既有方法的組合 ——
用 gappy POD（Everson & Sirovich 1995, JOSA A）從 K=100 sensor 在 t=0 建 divergence-free 場，之後自由積分、
不再 assimilate 任何資料，即 data assimilation 領域的 open-loop / free-run 對照組；且它用了 200 個 DNS
snapshots offline 建 rank-40 basis —— 比 PI-CON（只看 sensor stream）多得多的資訊。
所以這不是公平的 matched-assimilation baseline，是誠實揭露 forward-CFD 的優勢。
若被問「這方法叫什麼」：gappy-POD initialisation + open-loop forward integration，forward-CFD 只是本文簡稱。
⚠️ 2026-07-18 改版：本頁原本是「Forward-CFD t=5 snapshot vs PI-CON t≳3.3 late-window mean」
的數字表 —— 兩欄取不同時間窗，且 PI-CON 那欄用的 late-window 剛好避開 warm-up，看起來像
挑對自己有利的窗（fresh-eyes 委員 review 直接點名此頁）。改為全窗軌跡圖後不需要挑窗：
兩個端點都報，方向自明。舊的「為何兩欄窗不同」辯護稿已刪除，別再照舊講。

主視覺＝發散軌跡圖（thesis/figures/results/forward_cfd_divergence.png，
scripts/plot_forward_cfd_divergence.py 產生）。

講法：指圖左側 ——「forward-CFD 的起點其實比我們好：u rel-L₂ 只有 5.2%，因為它離線用了
200 張 DNS snapshot 建 POD-rank-40 基底。」再指右側 ——「open-loop 積分 5 秒後，chaotic
amplification 把它放大 29 倍到 152.8%；而 PI-CON 起點差（26.9%，IC warm-up）卻一路收斂
到 7.28%，因為它全程 re-condition on the sensor stream。兩條軌跡在 t≈2 交叉。」
這就是「為何不能直接 forward CFD」的完整答案：不是它一開始就爛，是它會發散。

⚠️ 圖的誠實性：橘色只有 t=0 與 t=5 **兩個實測點**，中間那條淡虛線是首尾連線、不是實測
軌跡（npz 只存這兩張場）。委員若問「中間長怎樣」→ 誠實答：沒有存中間快照，只能說端點。
不可宣稱那條線是量到的。

右下卡是 KE-as-misleading 的最強證據（呼應 §Conclusion ④）：t=5 時 forward-CFD 的
KE 只差 −3.85%、enstrophy +3.46%、能譜在 k∈[1,120] 內差 ≈10% —— 統計量全部「看起來沒事」，
但場已經完全去相關（u 152.8%）。σ_u/σ_v 從 DNS 的 2.32 掉到 0.90 是額外一擊：連
second-order statistic 都偏了。它留在 attractor 上，但相位錯了。

σ_u/σ_v = 0.90 是額外一擊：forward-CFD 連 Kolmogorov 流的各向異性都弄丟了
（DNS 2.32、PI-CON 2.30），所以它不只是「統計對、相位錯」，連 second-order statistic
都偏了。

數字來源：forward-CFD 端點與統計量出自 reports/forward_cfd_baseline_T5_rank40.json
（metrics_at_t0 / metrics_at_T，可從同名 .npz 重算驗證）與 appendix07.tex:106-119；
PI-CON 端點為 artifacts/exp245_seeds/eval_245_seed{a..e}_final 的 n=5 mean。
算術：152.8/5.2 = 29.3 → 29×（forward-CFD 發散）；7.28/26.9 = 0.27（PI-CON 收斂）；
同一時刻 t=5 的 pointwise 對比 152.8/7.28 = 21×。
（舊的「KE 2.4× vs pointwise 21×」對比已隨頁面改版移除：PI-CON 的 late-window KE 1.62%
不再出現在頁面上，那個比值現在無對應數字，別再講。）

⚠️ 可重現性缺口（2026-07-18 發現，尚未處理）：產生 forward-CFD 資料的 solver
`forward_cfd_baseline.py` **不在 repo，也不在 git 全歷史**，只留下 .npz/.json 產物。
現有數字可從 .npz 重算驗證（我核對過 152.8/203.9 完全吻合），但無法重跑或改變設定
（例如補存中間快照）。委員若問「這個 baseline 怎麼跑的」→ 可答方法（POD-rank-40 從
200 張 DNS snapshot 建基底 + ETDRK4 積分，dt=2.5e-4，20000 步，見 json metadata），
但不宜宣稱可立即重現。

⚠️ 必講的 caveat（appendix07:85, 100 明載）：這是 open-loop forecast reference，
不是 matched assimilation baseline；且 forward-CFD 另外用了 200 張 DNS snapshot
離線建 POD 基底 —— 它拿的資訊比 PI-CON 多，pointwise 仍崩掉。不可宣稱這是公平對打。
-->


---

<NavBar active="results" />

<SectionTag>§ Results · vs classical sensor-only interpolation</SectionTag>

# Kinetic energy error versus pointwise accuracy

<style>
.fb { width: 100%; border-collapse: collapse; font-size: 0.98rem; margin-top: 8px; }
.fb th { text-align: right; font-weight: 700; color: #6B7280; font-size: 0.86rem; text-transform: uppercase;
         letter-spacing: 0.04em; padding: 0 10px 8px 10px; border-bottom: 1px solid #D8D2E0; }
.fb th.m { text-align: left; }
.fb td { padding: 8px 10px; border-bottom: 1px solid #F1EDF5; color: #374151; text-align: right;
         font-variant-numeric: tabular-nums; }
.fb td.m { text-align: left; color: #1F1B2E; white-space: nowrap; }
.fb tr.ours td { background: #F7EDF8; border-bottom: none; font-weight: 700; }
.fb .win { color: #7F1084; font-weight: 700; }
.fb .trap { color: #E97132; font-weight: 700; }
.setup { display:grid; grid-template-columns:max-content 1fr; column-gap:14px; row-gap:6px;
         margin-top:14px; font-size:.80rem; line-height:1.3; color:#374151; }
.setup .n { color:#7F1084; font-weight:700; white-space:nowrap; }
</style>

<table class="fb">
  <thead>
    <tr>
      <th class="m">Method &nbsp;<span style="font-weight:400; text-transform:none; letter-spacing:0;">(same K = 100 sensors, no DNS access)</span></th>
      <th>KE %<br/><span style="font-weight:400; text-transform:none; letter-spacing:0; color:#E97132;">lower ≠ better</span></th><th>u L₂ %</th><th>v L₂ %</th><th><span class="raw">ω</span> L₂ %</th>
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
    <tr class="ours">
      <td class="m">PI-CON <span style="color:#9CA3AF; font-weight:400;">(ours, n = 5)</span></td>
      <td>5.71</td><td class="win">13.65</td><td class="win">17.52</td><td class="win">41.79</td>
    </tr>
  </tbody>
</table>

<div class="setup">
<div class="n">RBF</div><div>multiquadric kernel fitted through all 100 sensor values; shape <span class="raw">ε</span> = 10, of order one sensor spacing</div>
<div class="n">IDW</div><div>weighted average of the 100 values, weight <span class="raw">∝</span> 1/d²; exact at the sensors, flat in between</div>
<div class="n">trig-LSQ</div><div>least squares on a divergence-free Fourier basis truncated at k<sub>max</sub> = 5, the K = 100 Nyquist scale</div>
</div>


<div class="foot mt-1">Shape parameters fixed from a-priori scales, not tuned against DNS. Source, appendix <span class="raw">tab:fair_baselines</span>.</div>

<FooterLogos />

<!--
[Classical interpolation · 1.5min] 數字全部出自 thesis/back/appendix07.tex tab:fair_baselines，未改動。
這是全論文唯一「同 sensor、同指標、同 Re」的數字並排比較 —— 與文獻三篇（Mo & Magri / Kelshaw /
Parfenyev）的比較是規格對照，不是數值比較（Re 差 7.7× 以上，並排會誤導）。

本頁論點與 Forward-CFD 頁同一條：KE 單獨看會排錯名次。
- RBF 5.08 / trig-LSQ 4.42 的 KE 都比我們的 5.71 低，但 u rel-L₂ 是我們的 ~2 倍。
- 原因（appendix07 原文）：classical interpolation 把速度往 inter-sensor mean 收縮，
  壓低總動能，KE ratio 因為錯的理由靠近 DNS。pointwise u/v L₂ 才揭穿。
- 三個 baseline 都是 sensor-only（訓練與推論皆不碰 DNS），與 PI-CON 同約束，所以公平。
- RBF 的 ε=10、trig-LSQ 的 k_max=5 都由 a-priori 尺度定（sensor 間距、K=100 Nyquist），
  非為了壓低 DNS error 而調 —— 這點被問「你是不是挑弱的對手」時要講。

⚠️ 2026-07-19 底部結論句已整行刪除 —— 表格自己看得出來（PI-CON 三個 L₂ 欄全紫、
   數值約為 trig-LSQ 的一半），數字用講的即可。要講的版本：
   「u rel-L₂ 相對 trig-LSQ 降 47 %、相對 IDW 降 74 %；v 和 ω 也降 34–72 %。」
   **注意 47/74 只是 u 一個場**（25.87→13.65、52.88→13.65，與 appendix07:78 原文一致），
   三場平均其實是 42 % / 65 %，口頭不可講成「三個場都降 47–74 %」。
   實測六值：vs trig-LSQ u 47.2 / v 45.2 / ω 34.1；vs IDW u 74.2 / v 71.8 / ω 49.0。

⚠️ 2026-07-19 另修兩處：
 (1) 底部結論句原寫「Pointwise u, v, ω → −47 % / −74 %」——**那兩個數字只是 u 一個場**
     （25.87→13.65 = 47.2%；52.88→13.65 = 74.2%，與 appendix07:78 原文一致）。
     三場平均其實是 42 % / 65 %，掛 u,v,ω 的標籤等於把三個場混在一起報，會被抓。
     已改成「u rel-L₂ → −47 % / −74 %」，並補 (v and ω: −34 % to −72 %) 避免被問「那另外兩個呢」。
     實測六個值：vs trig-LSQ u 47.2 / v 45.2 / ω 34.1；vs IDW u 74.2 / v 71.8 / ω 49.0。
 (2) 三個方法原本只在名字後面掛一個裸參數（ε=10 / p=2 / k_max=5），沒說方法本身怎麼運作。
     已改成表格下方三行 setup，各一句講清楚做什麼 + 參數的物理來源。
     參數不是調出來的這件事從 notes 移到 slide 上（原本只有我知道，委員看不到）。

⚠️ appendix07 另有紀錄：trig-LSQ 在 k_max=8/12、RBF 在壞長度尺度下，誤差會大好幾個數量級。
   那些不穩定變體「excluded from the main comparison table」—— 我們選的是它們的強項組態，不是弱項。
   委員若問「為何只列這三列」，照此答。

⚠️ 這三個是通用經典方法，不是某篇論文的方法，appendix07 也沒有 \cite。不要說成「我們贏了某某論文」。
-->

---

<NavBar active="results" />

<SectionTag>§ Results · sensor count axis (O2)</SectionTag>

# K-scaling — cutoff vs. sensor count

<div class="text-sm mt-1" style="color:#374151;">
Where PI-CON departs from DNS follows the sensor Nyquist <span class="raw">k<sub>max</sub> = √(K/π)</span> — higher fidelity comes from <b style="color:#7F1084;">bandwidth expansion</b>, not architecture search.
</div>

<Card style="padding: 0.45rem 0.6rem;" class="mt-2">
<img :src="'/images/spectrum_k_scaling_triptych.png'"
     style="display: block; margin: 0 auto; max-width: 94%; height: auto;" />
</Card>

<div class="mt-2 text-xs leading-snug" style="color:#6B7280;">
Single-seed at the final protocol, read as a trend, not a fit.
</div>

<FooterLogos />

<!--
[K-scaling · 1.5min] 圖下小字精簡後的完整口徑（原註記字太小已縮）：
  K=100 是 seed-42 單 run（n=5 mean 為 5.71%）；K=400 run 用 512 collocation points（非 1024）。
  所以這條 K-scaling 是 single-seed trend，不是 fit。
O2 數量軸。主視覺＝三連能譜（scripts/plot_spectrum_k_scaling_triptych.py，
投影片專用；論文用 fig:k_scaling_spectra 的三張 subfigure）。講法：指綠線 —— 5.64 → 7.98 →
11.28 一路右移，而藍色 PI-CON 正好在綠線處脫離黑色 DNS，三格都是。這就是 chapter04:169
「the reconstruction bandwidth tracks the sensor-count Nyquist scale」。
KE 5.90 / 2.47 / 1.76 % 已標進 panel 標題（出處 tab:k_scaling_nyquist, chapter04.tex:285）。

也是 spectral-bias 反駁：若 cutoff 來自模型的 spectral bias，加 sensor 不會讓它右移；
它右移了，所以限制是 sensor 資訊量而非架構。

⚠️ 舊版用 KE 長條圖是錯的證據形式：主張是頻寬/冪律，長條圖畫不出斜率。實測 log-log
局部斜率 −1.26 (K=100→200) 與 −0.49 (K=200→400)、整體擬合 −0.87，三點不在一直線上；
若改畫 log-log 會當場暴露。頻譜圖才是論文真正的主張。

⚠️ 舊版寫「ratios (0.42, 0.30) follow the 1/K prediction within 20%」比論文更硬且丟了
caveat。chapter04:318 原文是「within 20% of this scaling estimate」，並明載「Because the
K=400 run uses fewer collocation points, the three-point curve should not be read as a
strict fit」。1/K 係由 k_max ∝ √K 推得（cutoff 以上未解析能量 ∝ k_max⁻² ∝ 1/K），屬
scaling estimate 非 prediction。兩個 caveat 已放回頁面（右卡）。被追問再給 ratio。

資料：EXP-269 (K=200)、EXP-270 (K=400, collo=512) 見 experiment_log_v2:493-494。
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
.nz tr.head td { color: #7F1084; }
.nz tr.head td.worst { color: #7F1084; background: #F7EDF8; }
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
<td>41.79</td><td>41.78</td><td>42.00</td><td>42.32</td>
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
[Noise robustness · 1.5min] O3 噪音軸。純數字頁 —— 五個點跨度僅 0.37 pp、全部通過門檻，
沒有可畫的形狀；折線圖只會是一條平線加一條碰不到的門檻線。

表格轉置：列＝指標、欄＝噪音強度。每一列由左往右讀就是一條變化軌跡；原本（列＝噪音、
欄＝指標）要直著讀才看得出變化。Δ 欄放最右並以分隔線切開，五個 + 號一眼看完。
門檻寫進 KE 的列標（target < 10），不另立卡片。

數據：EXP-290 final protocol n=5（experiment_log_v2:1507-1514）+ EXP-245 clean（:558）；
對應 thesis chapter04.tex:438 表與 :443。

講法：先掃 Δ 欄 —— 每一列都是正的，noise 不是免費的，div ratio 動最多 (+17.7 %)；
再看 KE 列的 10 % 格 6.08 —— 離門檻還有 3.9 pp。先承認再劃線，比宣稱「highly robust」強。

⚠️ 只有 KE 有 seed 標準差（log 僅記 KE mean/std，u/v/ω/div 只有 mean），故 ± 不入表格，
改以底注給範圍。0 % → 1 % 的 +0.035 pp 小於該散布，不可宣稱可分辨；唯 0 → 10 % 的
+0.37 pp 大於散布。
⚠️ ω rel-L₂ 非嚴格單調（1 % 時 41.790 → 41.781 微降 0.009，遠在 baseline std ±0.12 內）
—— 表上看得到，講的時候用論文原詞「monotone in the aggregate」(chapter04:443)，
不可說 strictly monotone。
⚠️ 不要用舊的 single-seed 10k noise 表（§6, EXP-258~261, KE 6.92→7.14）；log:1516 明載
已被 EXP-290 取代，chapter04:443 稱其為 older single-seed regularisation artefact。
-->


---

<NavBar active="results" />

<SectionTag>§ Results · engineering applicability</SectionTag>

# What K = 100 sensors support

<style>
/* 兩欄的寬度 = 波數帶的分割比例，截止線因此一路貫穿到底 —— 帶子是結構，不是裝飾。
   顏色兩個意思：紫 = 截止線以內（可解析）, 橘 = 截止線以外（觀測不到）。 */
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
.col h4 { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px; }
.col .row { font-size: 0.9rem; line-height: 1.5; color: #374151; }
.col .row b { font-weight: 700; }
.ar { color: #C9C6D0; font-weight: 400; }
</style>

<div class="sp band mt-3">
  <div class="lo"><span class="lbl" style="color:#7F1084;">RESOLVED, 98.9 % of the energy</span></div>
  <div class="hi"><span class="lbl" style="color:#9CA3AF;">UNOBSERVED</span></div>
</div>
<div class="sp kx">
  <span>k = 1</span>
  <span style="padding-left:6px;"><b>k<sub>max</sub> = √(K/π) = 5.64</b> <span class="ar">→</span> sensor Nyquist scale — a sensor budget, not an architecture</span>
</div>

<div class="sp">

<div class="col">
<h4 style="color:#7F1084;">Supported</h4>
<div class="row"><b style="color:#7F1084;">KE &amp; mean-flow monitoring</b><br/><span class="ar">→</span> 5.71 ± 0.12 %</div>
<div class="row mt-2"><b style="color:#7F1084;">Phase-locked control</b> @ k<sub>f</sub><br/><span class="ar">→</span> amp 0.99, phase ≲ 0.09 rad</div>
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
[Engineering applicability · 2min] 左卡：K=100 可支援的 use case — KE & mean-flow monitoring (5.71 ± 0.12%)、phase-locked control (forcing mode amplitude/phase recovered)、incompressibility check (resolved-bandwidth FD floor)、streaming deployment (filtering mode)。右卡：不適用 case — small-scale turbulence stats、fine vorticity filaments、acoustic/shock localisation 需多模態。Inference cost 移到下一頁獨立比較。
-->

---

<NavBar active="summary" />

<SectionTag>§ Conclusion · contributions</SectionTag>

# Contributions

<style>
.ct { display: grid; grid-template-columns: max-content 1fr; column-gap: 18px; row-gap: 0; margin-top: 10px; }
.ct .num { font-size: 1.4rem; font-weight: 700; color: #7F1084; line-height: 1; padding: 11px 0; }
.ct .body { padding: 9px 0; border-bottom: 1px solid #F1EDF5; }
.ct .ttl { font-size: 1.05rem; font-weight: 700; color: #1F1B2E; }
.ct .det { font-size: 0.95rem; color: #6B7280; margin-top: 4px; line-height: 1.4; }
.ct .det b { color: #7F1084; }
.ct .ob { font-size: 0.66rem; font-weight: 700; color: #16A34A; white-space: nowrap; padding: 8px 0; letter-spacing: 0.04em; }
.ar { color: #9CA3AF; font-weight: 400; }
</style>

<div class="ct">

<div class="num">①</div>
<div class="body">
<div class="ttl">PI-CON — a sparse-sensor inverse operator</div>
<div class="det">K = 100, Re = 10⁴, sensor + PDE only <span class="ar">→</span> KE <b>5.71 ± 0.12 %</b></div>
</div>

<div class="num">②</div>
<div class="body">
<div class="ttl">Sensing study — count, placement, noise</div>
<div class="det">Count sets resolution; placement and noise set reliability</div>
</div>

<div class="num">③</div>
<div class="body">
<div class="ttl">Cross-Reynolds feasibility</div>
<div class="det">Re = 10⁶ <span class="ar">→</span> KE <b>6.10 %</b> <span style="color:#9CA3AF;">(single seed)</span></div>
</div>

</div>

<div class="mt-3 px-3 py-2 rounded" style="background: rgba(127,16,132,0.06); border-left: 3px solid #7F1084;">
<div style="color:#374151; font-size: 0.95rem;">
Eight sensing configurations, KE <b>1.76 – 7.95 %</b> <span class="ar">, </span> all within the 10 % target
</div>
</div>

<FooterLogos />

<!--
[Contributions · 1.5min] 對應 thesis §5.2 的四條（chapter05.tex:18-21）。
2026-07-16 精簡：頁面只留「一句標題 + 一行結果」，細節數字改口述。

口述（頁面已移除，被問才給）：
① cross-attention 是 dominant standalone lever，CfC 透過 interaction 生效（2×2 分解：
   cross-attn −1.21、CfC +0.99、interaction −2.31、total −2.53 pp；p = 3.0×10⁻⁷）
② 數量：K 100 → 400 使 KE 5.90 → 1.76 %；位置：placement σ ≈ 6× training σ；
   噪音：10 % → KE 6.08 %。三軸皆在 10 % 門檻內。
③ Re = 10⁶ 用 K = 200，KE 6.10 % ≈ Re = 10⁴ baseline。chapter05:20 明載
   「single-seed and should be treated as an extension, not a multi-seed benchmark」——
   務必主動說是 extension 不是 benchmark。
④ PI-CON 把 pointwise u rel-L₂ 相對 classical interpolation 降低 47–74 %
   （chapter04:219, Appendix app:fair_baselines）。這條是次要貢獻，不與前三條等重。

舊版每張卡都是「一句敘述 + 一句數字」雙層結構，敘述那層是講的時候要說的話，不是要印的。
壓成單行標題 + 單行數字（147 → 約 70 字）。

數字出處：① tab:main_metrics；② tab:k_scaling_nyquist + chapter04:45 的 σ 比 6.2×；
③ tab:re1e6（chapter05:20 明載「single-seed and should be treated as an extension,
not a multi-seed benchmark」，故 ③ 保留該但書）；④ chapter04:219 的 47–74 %
（Appendix app:fair_baselines）。

⚠️ 2026-07-19：④「KE alone mis-ranks」已從本頁移除（原本灰階列為次要貢獻）。
   thesis chapter05 仍是四條，投影片只列三條 —— 被問「論文寫四條為何只講三條」時答：
   第四條是前三條的判讀方式，已在 §Results「Lower KE does not mean a better field」
   那頁完整呈現（RBF 5.08 / trig-LSQ 4.42 的 KE 比我們低但 u rel-L₂ 差近兩倍），
   不另立為並列貢獻。**不要說「論文沒有這條」——論文有。**
-->

---

<NavBar active="summary" />

<SectionTag>§ Conclusion · limitations and future work</SectionTag>

# Limitations and future work

<style>
.lx { width: 100%; border-collapse: collapse; margin-top: 16px; }
.lx th { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
         padding: 0 8px 4px 8px; border-bottom: 1px solid #D8D2E0; text-align: left; }
.lx td { padding: 13px 8px; border-bottom: 1px solid #F1EDF5; vertical-align: top; }
.lx .lim { font-size: 1.0rem; color: #1F1B2E; font-weight: 600; width: 42%; }
.lx .lim span { font-weight: 400; color: #9CA3AF; font-size: 0.92em; }
.lx .arw { color: #C9C6D0; width: 12px; padding: 13px 0; }
.lx .fix { font-size: 1.0rem; color: #374151; }
.lx .fix b { color: #7F1084; }
</style>

<table class="lx">
<thead><tr><th style="color:#E97132;">Limitation</th><th></th><th style="color:#7F1084;">Future work</th></tr></thead>
<tbody>
<tr>
<td class="lim">Uniform sensor clock</td>
<td class="arw">→</td>
<td class="fix">Irregular-clock test on the existing frames</td>
</tr>
<tr>
<td class="lim">Gaussian noise only</td>
<td class="arw">→</td>
<td class="fix">Bias, drift, dropout, calibration</td>
</tr>
<tr>
<td class="lim">Periodic domain, single forcing</td>
<td class="arw">→</td>
<td class="fix">Wall-bounded geometries</td>
</tr>
<tr>
<td class="lim">Cross-Re single seed</td>
<td class="arw">→</td>
<td class="fix">Multi-seed at Re = 10⁶</td>
</tr>
<tr>
<td class="lim">CFD-rigour gaps</td>
<td class="arw">→</td>
<td class="fix">Reynolds stresses, TKE budget, 4D-Var baseline</td>
</tr>
</tbody>
</table>

<div class="mt-4" style="color:#374151; font-size: 0.95rem;">
Validated scope <span style="color:#C9C6D0;">, </span> K = 100 <span style="color:#C9C6D0;">, </span> Re = 10⁴ <span style="color:#C9C6D0;">, </span> 2-D periodic Kolmogorov
</div>

<FooterLogos />

<!--
[Limitations & future work · 1.5min] 對應 thesis §5.3 + §5.4。原本拆兩頁（限制、未來工作），
2026-07-16 合併：每條限制的答案就是對應的未來工作，拆開才需要 FW①↔LIM② 互指標籤。

頁面只印 5 條最重要的；thesis §5.3 實際有 7 條，以下**兩條刻意不印，被問要答得出來**：
- K-scaling 是三點趨勢，非嚴格擬合：K=400 用 512 collocation 而非 1024
  （chapter04.tex:276 caption；chapter04:318 原文「should not be read as a strict fit」）。
  實測 log-log 局部斜率 −1.26 (K=100→200) 與 −0.49 (K=200→400)，三點不在一直線上。
  補法：matched-budget sweep K = 50/100/200/400。
- Per-case fitting，無 cross-case generality（chapter05:45）。補法：一個 operator 跨多場景 +
  training–simulation crossover（重訓比直接解流場貴時，operator 就沒有工程理由）。
另 thesis 尚有 Acceptance metrics / Diagnostic boundaries 兩條，屬細節。

口述（頁面已精簡掉的補述）：
- 「Uniform sensor clock」：CfC 的存在理由就是讀不規則時鐘，但 benchmark 均勻取樣 →
  這個能力從未被實驗觸及。補法最便宜：既有 201 frames 加時間軸 mask，對照 GRU + Δt，
  不需新 DNS。這也直接檢驗論文標題的「Continuous-Time」。
- 它同時解釋主結果頁 CfC 單獨 +0.99 pp：均勻時鐘下 Δt 為定值，chapter02.tex:212 的
  Δt 閘門吃不到變異，CfC 退化為多帶參數的 gated RNN，變差是預期而非意外。
- 「Periodic domain」：cylinder wake 已有初步驗證（Appendix），非全新領域。

⚠️ 「Uniform sensor clock」這條 thesis §5.3 原本沒有，2026-07-16 查證時發現 chapter05.tex:43
已有對應的 \textbf{Temporal sampling} 條目（先前 note 說「論文沒有」是過時資訊）。
-->
