<!--
  <TagChip>SciML</TagChip>
  Small inline chip for skill tags. Tinted purple to match brand.
-->
<template>
  <span class="tag-chip">
    <slot />
  </span>
</template>

<style scoped>
.tag-chip {
  display: inline-block;
  padding: 0.2rem 0.6rem;
  font-size: 0.90rem;
  border-radius: 0.35rem;
  background: var(--color-primary-tint, rgba(127, 16, 132, 0.08));
  color: var(--color-primary, #7F1084);
  font-weight: 600;
}
</style>
