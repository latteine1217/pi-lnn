<!--
  <MetricHero value="7.80%" label="kinetic-energy relative error" />
  <MetricHero value="6.89e-2" label="u-velocity RMSE" color="navy" size="md" />

  Big number + caption. Use for headline metrics.

  Props:
    value : string  — required
    label : string  — required
    color : 'accent' | 'navy'  (default 'accent')
    size  : 'sm' | 'md' | 'lg'  (default 'lg')
-->
<template>
  <div>
    <div class="metric-hero font-bold" :class="[colorClass, sizeClass]">
      {{ value }}
    </div>
    <div class="text-xs opacity-60 mt-1">{{ label }}</div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  value: { type: String, required: true },
  label: { type: String, required: true },
  color: { type: String, default: 'accent' },
  size:  { type: String, default: 'lg' },
})

const colorClass = computed(() =>
  props.color === 'navy' ? 'text-[#0F2D52]' : 'text-[#D97757]'
)

const sizeClass = computed(() => ({
  sm: 'text-xl',
  md: 'text-2xl',
  lg: 'text-4xl',
}[props.size] || 'text-4xl'))
</script>

<style scoped>
.metric-hero {
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.04em;
  line-height: 1;
}
</style>
