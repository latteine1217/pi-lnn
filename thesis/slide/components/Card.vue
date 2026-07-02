<!--
  <Card>...content...</Card>
  White rounded card with subtle border and padding.
  Background slightly translucent so the brand light-cyan ground bleeds through.
-->
<template>
  <div class="card">
    <slot />
  </div>
</template>

<style scoped>
.card {
  background: rgba(255, 255, 255, 0.86);
  border: 1px solid var(--color-border, #E5E0EC);
  border-radius: 0.75rem;
  padding: 1rem 1.2rem;
  backdrop-filter: blur(2px);
}
</style>
