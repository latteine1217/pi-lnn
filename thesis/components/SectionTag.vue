<!--
  <SectionTag>About me</SectionTag>
  Small uppercase tag with leading line accent — used at top of each slide.
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
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: #D97757;
  font-weight: 600;
  margin-bottom: 0.5rem;
}
.section-tag::before {
  content: '';
  display: inline-block;
  width: 1.2rem;
  height: 1px;
  background: #D97757;
}
</style>
