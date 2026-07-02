<!--
  <TagChip>SciML</TagChip>
  Small inline chip for skill tags.
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
  font-size: 0.72rem;
  border-radius: 0.35rem;
  background: rgba(15, 45, 82, 0.08);
  color: #0F2D52;
  font-weight: 500;
}
</style>
