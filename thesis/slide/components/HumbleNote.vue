<!--
  <HumbleNote>
    I haven't worked on these processes directly — but the underlying
    problem structure transfers cleanly. I'd love to learn the domain
    from the team.
  </HumbleNote>

  Centered, slightly muted "honest caveat" line. Used to soften strong claims
  (e.g. industrial mapping) and signal intellectual honesty.
-->
<template>
  <div class="humble-note">
    <span class="prefix">To be honest:</span>
    <slot />
  </div>
</template>

<style scoped>
.humble-note {
  text-align: center;
  font-size: 0.875rem;
  margin-top: 1.5rem;
}
.prefix {
  opacity: 0.6;
  margin-right: 0.25rem;
}
</style>
