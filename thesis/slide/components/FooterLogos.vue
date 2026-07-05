<!--
  <FooterLogos />

  底部一字排開的 logo bar，仿照 Group Meeting.pptx 母片：
  AC&T Lab · NTHU 校徽 · 系所文字 · 學院 logo · 國立清華大學文字 · ISE 文字。
  圖檔放在 /public/brand/，由 master image 抽出後縮圖另存。
-->
<template>
  <div class="footer-logos">
    <div class="logo-row">
      <img src="/brand/act_lab.png"        alt="AC&T Lab"   class="logo logo-actlab" />
      <img src="/brand/college_logo.png"   alt="College"    class="logo logo-college" />
      <img src="/brand/dept_wordmark.png"  alt="Dept"       class="logo logo-dept" />
      <div class="spacer" />
      <img src="/brand/nthu_seal.png"      alt="NTHU seal"  class="logo logo-seal" />
      <img src="/brand/nthu_wordmark.png"  alt="NTHU"       class="logo logo-nthu" />
    </div>
  </div>
</template>

<style scoped>
.footer-logos {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 60px;
  pointer-events: none;
  z-index: 4;
  background: linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.6) 30%, rgba(255,255,255,0.85) 100%);
}

.logo-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 100%;
  padding: 2px 20px 12px 20px;
  gap: 12px;
}

.spacer { flex: 1 1 auto; }

.logo {
  display: block;
  object-fit: contain;
  height: auto;
}

.logo-actlab  { max-height: 48px; }
.logo-college { max-height: 50px; }
.logo-dept    { max-height: 21px; }
.logo-seal    { max-height: 48px; }
.logo-nthu    { max-height: 21px; }
</style>
