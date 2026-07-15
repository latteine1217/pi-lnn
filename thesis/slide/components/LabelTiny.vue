<!--
  <LabelTiny>EDUCATION</LabelTiny>
  Tiny uppercase label, used inside cards.

  注意：text-transform: uppercase 會把小寫希臘字母轉成大寫（ω→Ω、π→Π），
  兩者在流體符號系統中意義不同。標籤內出現數學符號時，用 .raw 包住：
  <LabelTiny>Nyquist <span class="raw">√(K/π)</span></LabelTiny>
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
/* 數學符號逃生口：維持原字形，不受 uppercase 影響 */
.label-tiny :deep(.raw) {
  text-transform: none;
}
</style>
