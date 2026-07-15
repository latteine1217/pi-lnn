<!--
  <LabelTiny>EDUCATION</LabelTiny>
  Tiny uppercase label, used inside cards.

  標籤內若有數學符號，用 .raw 包住（uppercase 會改寫小寫希臘字母）：
  <LabelTiny>Nyquist <span class="raw">√(K/π)</span></LabelTiny>
  .raw 的豁免規則在 style.css 全域宣告，此處不重複。
-->
<template>
  <div class="label-tiny">
    <slot />
  </div>
</template>

<style scoped>
.label-tiny {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #6B7280;
  font-weight: 700;
  margin-bottom: 0.25rem;
}
</style>
