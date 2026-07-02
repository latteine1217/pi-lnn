<!--
  <Card>...content...</Card>
  White rounded card with subtle border and padding.
-->
<template>
  <div class="card">
    <slot />
  </div>
</template>

<style scoped>
.card {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1rem 1.2rem;
}
</style>
