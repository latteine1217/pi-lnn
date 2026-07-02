<!--
  <MappingRow
    target="Furnace temperature monitoring"
    note="few pyrometers, no internal sensors"
  />
  <MappingRow target="..." note="..." tag="exact match" />

  One row showing a target item with a contextual note and a small right-side tag.
  Default tag is "analogous problem" — tone-down language preferred over "structurally identical".

  Props:
    target : string  — bold navy headline
    note   : string  — small grey caption
    tag    : string  — small right-aligned label (default: 'analogous problem')
-->
<template>
  <div class="mapping-row">
    <div>
      <div class="mapping-target">{{ target }}</div>
      <div class="mapping-note">{{ note }}</div>
    </div>
    <div class="mapping-tag">{{ tag }}</div>
  </div>
</template>

<script setup>
defineProps({
  target: { type: String, required: true },
  note:   { type: String, required: true },
  tag:    { type: String, default: 'analogous problem' },
})
</script>

<style scoped>
.mapping-row {
  background: #fff7ed;
  border: 1px solid #fdba74;
  border-radius: 0.5rem;
  padding: 0.5rem 0.75rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.mapping-target {
  font-size: 0.875rem;
  font-weight: 600;
  color: #0F2D52;
}
.mapping-note {
  font-size: 10px;
  opacity: 0.6;
}
.mapping-tag {
  color: #D97757;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
}
</style>
