<!--
  <NavBar active="results" />

  頂部 chevron tab 導引列，仿照 Group Meeting.pptx 母片的章節導引列。
  active = 'background' | 'literature' | 'motivation' | 'objective' | 'method'
         | 'results' | 'summary' | null
  active 的 tab 以紫底白字呈現，其他 tab 灰底紫字。

  2026-07-18（指導教授要求）：於 Background 與 Objective 之間補上 Literature review，
  並新增 Motivation 分段（數頁由 Background 改隸此段）。tab 由 5 個增為 7 個 —— 標籤
  縮寫為 "Literature" / "Motivation" 以免 980px 寬度下擠壓換行。
-->
<template>
  <div class="nav-bar">
    <div
      v-for="(t, i) in tabs"
      :key="t.id"
      class="tab"
      :class="{ active: active === t.id, first: i === 0 }"
    >
      <span>{{ t.label }}</span>
    </div>
  </div>
</template>

<script setup>
defineProps({
  active: { type: String, default: null },
})

const tabs = [
  { id: 'background', label: 'Background' },
  { id: 'literature', label: 'Literature' },
  { id: 'motivation', label: 'Motivation' },
  { id: 'objective',  label: 'Objective'  },
  { id: 'method',     label: 'Methodology' },
  { id: 'results',    label: 'Results'    },
  { id: 'summary',    label: 'Summary'    },
]
</script>

<style scoped>
.nav-bar {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 36px;
  display: flex;
  z-index: 5;
  pointer-events: none;
}

.tab {
  flex: 1 1 0;
  position: relative;
  height: 36px;
  display: flex;
  align-items: center;
  /* 7 個 tab 均分 980px（每個約 140px），故較 5-tab 版縮小內距與字級、
     並收緊字距，讓 "Methodology" 這種長標籤仍能單行呈現。 */
  padding-left: 0.75rem;
  background: rgba(127, 16, 132, 0.12);
  color: var(--color-primary, #7F1084);
  font-family: Arial, 'Helvetica Neue', sans-serif;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  white-space: nowrap;
  overflow: hidden;
}

/* chevron 右側尖角 — 使用 clip-path 模擬 PowerPoint Pentagon/Chevron */
.tab {
  clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 50%, calc(100% - 14px) 100%, 0 100%);
  margin-right: -10px;
}

.tab.first {
  /* Pentagon — 左側直角、右側尖角 */
  clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 50%, calc(100% - 14px) 100%, 0 100%);
}

.tab.active {
  background: var(--color-primary, #7F1084);
  color: #ffffff;
  z-index: 2;
}
</style>
