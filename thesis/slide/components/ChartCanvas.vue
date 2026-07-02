<!--
  <ChartCanvas
    type="line" | "bar" | "scatter"
    :data="chartJsData"
    :options="chartJsOptions"
    height="280px"
  />

  Generic chart.js wrapper. chart.js is registered globally in setup/main.ts.
  Brand defaults (navy text, coral accent, no animation) are applied in main.ts.
  This component only adds responsive sizing + auto-cleanup.
-->
<template>
  <div class="chart-wrap" :style="{ height }">
    <canvas ref="canvas" />
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, watch, nextTick } from 'vue'
import { Chart } from 'chart.js'

const props = defineProps({
  type:    { type: String, required: true },
  data:    { type: Object, required: true },
  options: { type: Object, default: () => ({}) },
  height:  { type: String, default: '280px' },
})

const canvas = ref(null)
let chart = null

// 顏色與 setup/main.ts 一致：NTHU 紫主色 / 橘 accent / 灰 muted。
const PRIMARY = '#7F1084'
const ACCENT  = '#E97132'
const MUTED   = '#6B7280'

const baseOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: { boxWidth: 14, boxHeight: 8, color: PRIMARY, font: { size: 11 } },
      position: 'top',
      align: 'end',
    },
    tooltip: {
      backgroundColor: PRIMARY,
      titleColor: '#fff',
      bodyColor: '#f3f4f6',
      borderColor: ACCENT,
      borderWidth: 1,
      padding: 8,
      titleFont: { size: 11 },
      bodyFont: { size: 11 },
    },
  },
  // Only style the x axis at the base layer. The y / y1 / y2 axes are
  // declared per-chart, since some charts use a single y while others use
  // dual y1/y2; merging a default `y` would create a phantom 0–1 axis on
  // dual-axis charts.
  scales: {
    x: { ticks: { color: MUTED, font: { size: 10 } }, grid: { color: '#f3f4f6', drawTicks: false } },
  },
  elements: {
    point: { radius: 2, hoverRadius: 4 },
    line:  { borderWidth: 1.8, tension: 0.0 },
  },
}

function deepMerge(a, b) {
  if (!b) return a
  const out = Array.isArray(a) ? [...a] : { ...a }
  for (const k of Object.keys(b)) {
    const av = a ? a[k] : undefined
    const bv = b[k]
    if (bv && typeof bv === 'object' && !Array.isArray(bv)) {
      out[k] = deepMerge(av || {}, bv)
    } else {
      out[k] = bv
    }
  }
  return out
}

async function build() {
  await nextTick()
  if (!canvas.value) return
  if (chart) { chart.destroy(); chart = null }
  chart = new Chart(canvas.value, {
    type: props.type,
    data: props.data,
    options: deepMerge(baseOptions, props.options || {}),
  })
}

onMounted(build)
onUnmounted(() => { if (chart) chart.destroy() })
watch(() => [props.type, props.data, props.options], build, { deep: true })
</script>

<style scoped>
.chart-wrap {
  position: relative;
  width: 100%;
}
</style>
