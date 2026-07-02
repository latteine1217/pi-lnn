<!--
  <Pill>Model design → Data → HPC → Tooling</Pill>
  Centered navy pill — used for punchline / takeaway summaries.
-->
<template>
  <div class="text-center">
    <div class="pill">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.pill {
  display: inline-block;
  background-color: #0F2D52;
  color: white;
  padding: 0.5rem 1.25rem;
  border-radius: 9999px;
  font-size: 0.85rem;
  font-weight: 500;
  letter-spacing: 0.02em;
}
</style>
