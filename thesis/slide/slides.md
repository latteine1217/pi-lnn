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
[PINN 怎麼運作 · 1min]
• Training：座標 → 預測 → data + PDE residual → 更新 θ
• Inference：θ 凍結，只送座標取值
⚠️ 這是 PINN 通論，不是 PI-CON
⚠️ 圖中層數寬度為示意
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
[傳統方法要什麼 · 1.5min]
• ROM 要先有完整流場 —— 開機前不存在
• DA 要 solver 進迴圈 —— Re=10⁴ 太慢
⚠️ RBF/IDW/trig-LSQ 是自建 baseline，不列文獻
⚠️ 4D-Var 沒實作 → 誠實答 future work
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

<div class="mt-4 px-4 py-3 rounded text-base leading-snug" style="background:rgba(127,16,132,.06); border-left:4px solid #7F1084;">
<b style="color:#7F1084;">Why this matters:</b> the reconstruction is explicitly conditioned on what the instruments measured, while remaining queryable at arbitrary space–time coordinates.
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
<div class="mt-1 text-xs leading-snug" style="color:#6B7280;">
<b style="color:#374151;">Wavenumber k</b> indexes the spatial Fourier transform of the field: low k = large vortices, high k = fine structures.
</div>
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
<thead><tr><th class="l">Sensors</th><th><span class="raw">k<sub>max</sub></span></th><th>DNS energy inside</th></tr></thead>
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
.resolution .formula { font-size:1.12rem; color:#7F1084; font-weight:700; text-align:center; margin:9px 0 9px; padding:7px 6px; background:#FAF2FB; border-radius:7px; }
.resolution .copy { font-size:.85rem; color:#374151; line-height:1.4; }
.resolution .evidence { display:grid; grid-template-columns:78px 1fr; column-gap:10px; row-gap:7px; align-items:baseline; margin-top:11px; padding-top:9px; border-top:1px solid #E5E0EC; }
.resolution .evidence b { color:#7F1084; font-size:.82rem; }
.resolution .evidence span { color:#374151; font-size:.80rem; line-height:1.3; }
.resolution .plot img { width:100%; height:270px; object-fit:contain; display:block; }
</style>

<div class="resolution">
  <div class="cardx">
    <div class="tiny">Mode-counting estimate</div>
    <div class="copy" style="margin-bottom:6px;"><b>Wavenumber <span class="raw">k</span></b> = spatial Fourier index. Low <span class="raw">k</span> = large vortices, high <span class="raw">k</span> = fine structures.</div>
    <div class="formula">k<sub>sensor</sub> ≡ √(K/π) = 5.64</div>
    <div class="copy">Equating K point measurements to the approximate number πk² of Fourier modes defines a <b>sensor-count scale</b>. It is not a recovery bound.</div>
    <div class="evidence">
      <b>Energy</b><span>98.9% of DNS kinetic energy lies below k<sub>sensor</sub>.</span>
      <b>k ≤ 5</b><span>well-conditioned measurement map, κ ≈ 7.</span>
      <b>k ≤ 8</b><span>still observable but ill-conditioned, κ ≈ 7×10².</span>
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
[研究目標 · 1.5min]
• O1 能力：KE < 10 % (n=5)／≥ 2 pp @ p<0.01／≥ 5× 快
• O2 數量：k_max = √(K/π) ≈ 5.64
• O3 位置噪音：影響 reliability，不影響 feasibility
• 橋接：「三個目標分別由架構、數量、位置噪音回答」
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
[Kolmogorov flow · 1.5min]
• L₀/U₀ 是 convention；λ_f、U_rms 才是流場解出的尺度
• Re_f = 2.51×10³ 與 Re₀ = 10⁴ 分開報
⚠️ 全簡報無因次宣告點 —— 之後一律不帶單位
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
<tr><td class="c">Resolution <span style="color:#9CA3AF;">[Pope 2000]</span></td><td class="m">k<sub>max</sub><span class="raw">η</span> = 7.61 / 1.91 <span style="color:#9CA3AF;">≥ 1.5</span></td><td class="j"><span class="ok">resolved</span></td></tr>
<tr><td class="c">Grid independence</td><td class="m">ΔKE ≤ 0.064 %</td><td class="j"><span class="ok">verified</span></td></tr>
<tr><td class="c">Time-step convergence</td><td class="m">Δu<sub>rel-L₂</sub> = 7.0×10⁻⁶</td><td class="j"><span class="ok">verified</span></td></tr>
<tr><td class="c">Solver divergence <span style="color:#9CA3AF;">(run grid)</span></td><td class="m">‖∇·u‖<sub>∞</sub> ≲ 10⁻¹²</td><td class="j"><span class="ok">round-off</span></td></tr>
<tr><td class="c">Turnover coverage</td><td class="m">T/t<sub>eddy</sub> = 2.51 <span style="color:#9CA3AF;">≥ 50 ideal</span></td><td class="j"><span class="warn">limited</span></td></tr>
</tbody>
</table>
<div class="cap2"><b style="color:#0F2D52;">Resolved, not statistically converged</b> — one DNS realisation, not an ensemble.</div>
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
• 五列驗證：Pope 7.61/1.91・grid 0.064 %・Δt 7e-6・div ≲1e-12・T/t_eddy 2.51 limited
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
<span class="k">Purpose</span><span class="v"><b style="color:#7F1084;">large-scale proxy for QR-pivot only</b></span>
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
[LES 佈點 · 1.5min]
• LES 只做佈點，DNS 仍是 reference
• 數值達標，統計窗未建立（4.9 < 10）
⚠️ closure 是 hyperviscosity 單獨用；Bardina 只在 low-fidelity 變體
⚠️ 不可宣稱 LES 收斂
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

  <rect x="190" y="25" width="160" height="65" fill="#0F2D52" stroke="#0F2D52" stroke-width="1.3"/>
  <text x="270" y="52" text-anchor="middle" fill="#FFF" style="font-size:15px;font-weight:700;">Fourier embed</text>
  <text x="270" y="75" text-anchor="middle" fill="#CFE0F2" style="font-size:11.5px;font-weight:600;">Nq × 128</text>

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
• 深藍＝沿用（含 Fourier embed）／橘＝新增（CfC、cross-attn）
• 兩條路徑 + 紫線 training loop
• PI-CON 名稱首次登場
⚠️ Fourier 是藍不是橘（B0 也有）
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

<style>
/* 兩條 attention 公式相鄰，KaTeX 自帶的 display margin 疊起來會頂出頁尾。
   只收這一頁的公式間距，不動別頁（Slidev 會把此 style scope 到本 slide）。 */
.slidev-page-13 .katex-display { margin: 0.15em 0 !important; }
</style>

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

<div class="grid grid-cols-2 gap-5 mt-3 text-sm">

<Card>
<LabelTiny>AUGMENTED LAGRANGIAN (AL) ON CONTINUITY</LabelTiny>

<div class="mt-2" style="font-size: 0.82em;">

$$\mathcal{L}_{\text{AL}} \;=\; \lambda\,\mathcal{L}_{\text{cont}} \;+\; \tfrac{\rho}{2}\,\mathcal{L}_{\text{cont}}^{\,2}$$

</div>

<div class="mt-1" style="font-size: 0.82em;">

$$\mathcal{L}_{\text{cont}} \,=\, \mathbb{E}_{\text{collocation}}\big[(\partial_x u + \partial_y v)^2\big] \,\ge\, 0$$

</div>

<div class="mt-1" style="font-size: 0.82em;">

$$\lambda \,\leftarrow\, \mathrm{clip}\big(\lambda + \rho\,\overline{\mathcal{L}}_{\text{cont}},\; 0,\; \Lambda_{\max}\big)$$

</div>

<div class="mt-3" style="display:grid; grid-template-columns:max-content 1fr; column-gap:12px; row-gap:5px;
     align-items:baseline; font-size:0.76rem; line-height:1.25;">
<b style="color:#7F1084;"><span class="raw">𝓛<sub>cont</sub></span></b><span style="color:#374151;">mean squared continuity residual on the collocation points</span>
<b style="color:#7F1084;"><span class="raw">λ</span></b><span style="color:#374151;">multiplier, raised by dual ascent every 100 steps</span>
<b style="color:#7F1084;"><span class="raw">ρ</span> = 0.1</b><span style="color:#374151;">penalty strength</span>
<b style="color:#7F1084;"><span class="raw" style="text-decoration:overline;">𝓛</span><span class="raw"><sub>cont</sub></span></b><span style="color:#374151;">EMA of <span class="raw">𝓛<sub>cont</sub></span>, <span class="raw">β</span> = 0.5, smooths the update</span>
<b style="color:#7F1084;"><span class="raw">Λ</span><sub>max</sub> = 10</b><span style="color:#374151;">safety clip <span style="color:#9CA3AF;">— <span class="raw">λ</span> settles near 0.39, far below it</span></span>
</div>

<div class="mt-2 pt-2 text-xs leading-snug" style="color:#6B7280; border-top:1px solid #E5E0EC;">
<b style="color:#E97132;"><span class="raw">𝓛<sub>cont</sub></span> ≥ 0, so <span class="raw">λ</span> never decreases</b> — an accumulated-multiplier schedule, not a textbook Lagrangian whose multiplier changes sign.
</div>
</Card>

<Card>
<LabelTiny>IS THE CONSTRAINT DOING ANYTHING?</LabelTiny>

<div class="mt-3 text-xs" style="color:#6B7280;">
Divergence ratio <span class="raw">‖∇·u‖₂ / ‖∇u‖<sub>F</sub><sup>DNS</sup></span> — residual divergence over the DNS velocity-gradient magnitude, same grid.
</div>

<div class="mt-2" style="display:grid; grid-template-columns:1fr max-content; column-gap:12px; row-gap:9px; align-items:baseline; font-size: 0.92rem; font-variant-numeric:tabular-nums;">
<span style="color:#6B7280;">DNS, full cascade</span><span style="color:#9CA3AF;">1.04 %</span>
<span style="color:#6B7280;">DNS, band-limited to k ≤ 16</span><span style="color:#9CA3AF;">0.38 %</span>
<span style="color:#1F1B2E; font-weight:600;">PI-CON, same bandwidth</span><span style="color:#7F1084; font-weight:700;">0.39 %</span>
</div>

<div class="mt-4 pt-3 text-xs leading-snug" style="color:#6B7280; border-top: 1px solid #E5E0EC;">
At the finite-difference floor of its resolved bandwidth — <b>not</b> below DNS.
</div>

</Card>

</div>

<FooterLogos />

<!--
[AL on continuity · 1.5min]
• 右卡三列就是「約束有沒有作用」的答案
⚠️ 問 SIMPLE/PISO（版上已無，只能口述）：
　 ① 它逐點投影，我們是平均意義　② 逐點不可微、進不了 GradNorm　③ 用三個數字答不用類比答
⚠️ 絕不可說 sub-DNS 或「比 DNS 更不可壓縮」
⚠️ Λ_max 從未撞到（λ 停在 0.386 = clip 的 3.9 %）
• 問 ρ → 拉到 1 可把 div 壓到 0.28 %，但犧牲場精度
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
[最佳化 · 2min]
• SOAP + SF：二階 + Polyak，chaotic 窄谷需要
• GradNorm：4-task 梯度等化，每 1000 步
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
[模型與訓練配置 · 1min]
• 只給別處沒有的：模型尺寸與訓練預算
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
<b style="color:#7F1084;">Invariant</b>,  DNS field never enters ℒ.
</div>
</Card>

</div>

<FooterLogos />

<!--
[誤差指標與 loss · 1.5min]
• 四個量：global rel-L₂／KE MAPE／逐時 RMSE／div ratio
• 底線：DNS field 從不進 L
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
<LabelTiny>Welch <span class="raw">t</span>-test &nbsp;<span class="opacity-60">5 random seeds per cell</span></LabelTiny>
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
[誤差的波數結構 · 2min]
• k ≤ 5 就是 k_max 5.64；越過後 κ 由 7 → 7×10²，加大網路補不回來
⚠️ 別說成「不可觀測」—— 那要到 k ≈ 8 才成立
-->

---

<NavBar active="results" />

<SectionTag>§ Results · EXP-245 baseline (B3 + LES_T50, 1024 collo)</SectionTag>

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
<th><span class="raw">u / v rel-L₂</span> (%)</th>
</tr>
</thead>
<tbody>
<tr class="main">
<td>LES T = 50 &nbsp;<span class="sub">dimensionless, main pipeline</span></td>
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
[佈點 · 2min]
• ① oracle 贏 KE、LES 贏 pointwise → **trade-off**　② −2.24 pp vs random　③ variance 6.4×
• 三者皆 < 10 % → O3「影響 reliability 不影響 feasibility」
⚠️ 不可拿 LES 能譜比 DNS 當品質證據
-->


---

<NavBar active="results" />

<SectionTag>§ Results · vs an open-loop free run (no data assimilated)</SectionTag>

# Trajectory divergence of the open-loop free run

<style>
.olp { display:flex; align-items:stretch; gap:8px; margin-top:10px; }
.olp .st { flex:1; background:rgba(255,255,255,.72); border:1px solid #E5E0EC; border-radius:7px;
           padding:8px 11px; font-size:.80rem; line-height:1.35; color:#1F1B2E; }
.olp .st .no { display:inline-block; width:16px; height:16px; border-radius:50%;
               background:#E97132; color:#fff; font-size:.66rem; font-weight:700;
               text-align:center; line-height:16px; margin-right:6px; }
.olp .st .sub { color:#6B7280; font-size:.73rem; }
.olp .ar { align-self:center; color:#C9C6D0; font-size:1rem; }
.olp .picon { flex:1; background:#FAF2FB; border:1px solid #C9A6CC; border-radius:7px;
              padding:8px 11px; font-size:.80rem; line-height:1.35; color:#1F1B2E; }
.olp .picon .no { background:#7F1084; }
.st5 { display:grid; grid-template-columns:1fr max-content; column-gap:12px; row-gap:10px;
       align-items:baseline; font-size:.95rem; font-variant-numeric:tabular-nums; margin-top:6px; }
.st5 .lab { color:#6B7280; }
</style>

<div class="olp">
  <div class="st"><span class="no">1</span><b>Sensors at <span class="raw">t</span> = 0 only</b><br><span class="sub">the same K = 100 values</span></div>
  <span class="ar">→</span>
  <div class="st"><span class="no">2</span><b>Gappy POD, rank 40</b><br><span class="sub">divergence-free start field</span></div>
  <span class="ar">→</span>
  <div class="st"><span class="no">3</span><b>Same DNS solver, free run</b><br><span class="sub">no sensor data after <span class="raw">t</span> = 0</span></div>
  <span class="ar" style="color:#7F1084;">vs</span>
  <div class="picon"><b style="color:#7F1084;">PI-CON</b><br><span class="sub">reads the sensors at <b style="color:#7F1084;">every</b> <span class="raw">t</span></span></div>
</div>

<div class="grid grid-cols-5 gap-4 mt-3">

<div class="col-span-3">
<Card style="padding: 0.5rem 0.7rem;">
<img :src="'/images/forward_cfd_divergence.png'" style="width: 100%; max-height: 196px; object-fit: contain;" />
<div class="text-[10px] leading-snug" style="color:#9CA3AF; margin-top:2px;">POD basis from <b>200 offline DNS snapshots</b> [Everson &amp; Sirovich 1995] — more than PI-CON ever sees.</div>
</Card>
</div>

<div class="col-span-2">
<Card style="padding-top: 0.9rem; padding-bottom: 0.9rem;">
<LabelTiny>At <span class="raw">t</span> = 5</LabelTiny>
<div class="st5">
<span class="lab">KE</span><span>−9.3 %</span>
<span class="lab">Enstrophy</span><span>−10.5 %</span>
<span class="lab">Cumulative spectrum</span><span>9.6 %</span>
<span style="color:#1F1B2E; font-weight:600;">The field itself</span><span style="color:#E97132; font-weight:700;">160 % wrong</span>
</div>
<div class="mt-4 pt-3 text-xs leading-snug" style="color:#6B7280; border-top:1px solid #E5E0EC;">
The forecast stays on the attractor — at the <b style="color:#E97132;">wrong phase</b>. Growth is chaotic sensitivity to the start field, <b>not solver error</b>.
</div>
</Card>
</div>

</div>


<FooterLogos />

<!--
[Open-loop free run · 2min]
• 三步流程：t=0 的 100 個量測 → gappy POD rank 40 補成無散度初始場 → 同一支 DNS solver 自由積分
• 我們唯一的差別：全程讀 sensor。指圖：兩線 t≈1 交叉，橘線 t=2.4 穿越 100 %，末端 160.3 %（30×）
• 右卡：bulk 只差 9–10 %，場卻錯 160 % → **17× 落差**，KE 會騙人
⚠️ 曲線是**重跑**（scripts/forward_cfd_baseline.py --integrate，201 幀）。原始 .npz 只存首尾兩點，
   solver 不在 git 歷史；重跑 5 項指紋中 4 項吻合、IC 殘差約 1 %。混沌放大使端點不同
   （u 160.3 vs 152.8、v 172.9 vs 203.9）→ **兩批不可混用**，appendix07 已同步改用重跑值
⚠️ 非 matched baseline —— POD 基底用了 200 張離線 DNS，比我們看到的多
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
    <tr class="ours">
      <td class="m">PI-CON <span style="color:#9CA3AF; font-weight:400;">(ours, n = 5)</span></td>
      <td>5.71</td><td class="win">13.65</td><td class="win">17.52</td><td class="win">41.77</td>
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
[經典內插對照 · 1.5min]
• 唯一「同 sensor・同指標・同 Re」的並排比較
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
[工程適用範圍 · 2min]
• 左＝支援：KE 監測／phase-locked control／不可壓縮檢查／streaming
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
Developed <b>PI-CON</b>, a CfC–DeepONet operator with cross-attention readout and an augmented-Lagrangian continuity constraint, trained on sensors and the NS residual alone.
</div>

<div class="num">2.</div>
<div class="body">
Reconstructed Re = 10⁴ Kolmogorov flow from <b>K = 100</b> sensors to <b>5.71 ± 0.12 %</b> KE error over five seeds, no DNS field in the training loss.
</div>

<div class="num">3.</div>
<div class="body">
Isolated the architectural gain by a matched-budget 2 × 2 ablation: 8.23 % <span style="color:#C9C6D0;">→</span> <b>5.71 %</b> <i>(p = 3 × 10⁻⁷)</i>, cross-attention dominant.
</div>

<div class="num">4.</div>
<div class="body">
Quantified the sensing configuration: count sets bandwidth <i>(K = 100 → 400 cuts KE error <b>70 %</b>)</i>; placement and 10 % noise <i>(6.08 %)</i> affect reliability, not feasibility.
</div>

<div class="num">5.</div>
<div class="body">
Showed cross-Reynolds feasibility at <b>Re = 10⁶</b> <i>(6.10 %, single seed)</i>; eight configurations all within <b>1.76 – 7.95 %</b>, under the 10 % target.
</div>

</div>

<FooterLogos />

<!--
[貢獻 · 2min]
• 五條各講「做了什麼 + 結果如何」
• 問「論文四條為何只講三點」→ 第四條是判讀方式，已在內插頁講過。**不可說論文沒有**
• Re=10⁶ 說 extension，不是 benchmark
⚠️ 數字出處：log:132／580／1561／208／385・chapter04:117／247
⚠️ 第 4 條用「降 70 %」避開 single-seed 與 n=5 混用
⚠️ 未裁決：p24 寫 −2.52，論文寫 −2.53
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
[限制與未來工作 · 1.5min]
• Uniform clock 要主動展開：CfC 的賣點從未被測；補法＝既有 frames 加時間 mask
　 它也解釋 CfC 單獨 +0.99 pp（Δt 定值 → 閘門吃不到變異）
• Periodic domain：cylinder 已有初步驗證
⚠️ 論文 7 條只印 5 條。備答：K-scaling 非嚴格擬合、per-case fitting 無跨場景泛化
-->
