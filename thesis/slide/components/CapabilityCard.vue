<!--
  <CapabilityCard
    icon="◆"
    label="HPC operations"
    headline="29-node SLURM cluster"
    :bullets="[
      '2 GPU nodes · InfiniBand · 16 TB NFS',
      'Unified scheduling for ML + CFD',
      'Ansible playbooks for reproducibility',
    ]"
    tag="github.com/your/repo"
  />

  Three-row card: icon+label / bold headline / bulleted detail / optional footer tag.
  Works for the "What I can do" capability matrix.

  Props:
    icon     : string
    label    : string  (uppercase tiny label)
    headline : string  (bold blue text)
    bullets  : string[] | undefined
    tag      : string | undefined  (footer URL/note)
-->
<template>
  <div class="card flex flex-col">
    <div class="flex items-center gap-2 mb-1">
      <div class="cap-icon">{{ icon }}</div>
      <div class="cap-label">{{ label }}</div>
    </div>
    <div class="cap-headline">{{ headline }}</div>
    <div v-if="bullets && bullets.length" class="mt-2 space-y-1 flex-1">
      <BulletRow v-for="b in bullets" :key="b">{{ b }}</BulletRow>
    </div>
    <div v-if="tag" class="cap-tag">{{ tag }}</div>
  </div>
</template>

<script setup>
defineProps({
  icon:     { type: String, required: true },
  label:    { type: String, required: true },
  headline: { type: String, required: true },
  bullets:  { type: Array,  default: () => [] },
  tag:      { type: String, default: '' },
})
</script>

<style scoped>
.card {
  background: rgba(255, 255, 255, 0.86);
  border: 1px solid var(--color-border, #E5E0EC);
  border-radius: 0.75rem;
  padding: 1rem 1.2rem;
}
.cap-icon  { font-size: 1.125rem; color: var(--color-accent, #E97132); }
.cap-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--color-muted, #6B7280);
  font-weight: 600;
}
.cap-headline {
  font-size: 1rem;
  font-weight: 700;
  color: var(--color-primary, #7F1084);
  line-height: 1.3;
}
.cap-tag {
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--color-border, #E5E0EC);
  font-size: 9px;
  opacity: 0.5;
  font-family: 'JetBrains Mono', monospace;
  word-break: break-all;
}
</style>
