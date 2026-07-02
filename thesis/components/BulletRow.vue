<!--
  <BulletRow>Multi-fidelity training</BulletRow>
  Compact "–" prefixed row for in-card lists.
-->
<template>
  <div class="bullet-row">
    <span><slot /></span>
  </div>
</template>

<style scoped>
.bullet-row {
  display: flex;
  gap: 0.5rem;
  align-items: baseline;
  font-size: 0.88rem;
  line-height: 1.45;
}
.bullet-row::before {
  content: '–';
  color: #D97757;
  flex-shrink: 0;
}
</style>
