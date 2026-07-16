<!--
  <SectionTag>About me</SectionTag>
  Small uppercase tag with leading line accent — used at top of each slide.
  Color tracks --color-primary (NTHU purple) by default.
-->
<template>
  <div class="section-tag">
    <slot />
  </div>
</template>

<style scoped>
.section-tag {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.83rem;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--color-primary, #7F1084);
  font-weight: 600;
  margin-bottom: 0.5rem;
}
.section-tag::before {
  content: '';
  display: inline-block;
  width: 1.2rem;
  height: 1px;
  background: var(--color-primary, #7F1084);
}
</style>
