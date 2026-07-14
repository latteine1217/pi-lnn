<!--
  <LabelTiny>EDUCATION</LabelTiny>
  Tiny uppercase label, used inside cards.
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
