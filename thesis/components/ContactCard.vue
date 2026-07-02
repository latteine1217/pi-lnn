<!--
  <ContactCard label="Email" value="felix.tc.tw@gmail.com" />

  Centered contact card — used in Q&A / Thank-you slide.
  No-wrap on the value to prevent ugly mid-string breaks.

  Props:
    label : string
    value : string
-->
<template>
  <div class="contact-card">
    <div class="contact-label">{{ label }}</div>
    <div class="contact-value">{{ value }}</div>
  </div>
</template>

<script setup>
defineProps({
  label: { type: String, required: true },
  value: { type: String, required: true },
})
</script>

<style scoped>
.contact-card {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1rem 1.2rem;
  text-align: center;
}
.contact-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #6B7280;
  font-weight: 600;
  margin-bottom: 0.25rem;
}
.contact-value {
  font-size: 11px;
  color: #0F2D52;
  font-family: 'JetBrains Mono', monospace;
  white-space: nowrap;
}
</style>
