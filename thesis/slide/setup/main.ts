import { defineAppSetup } from '@slidev/types'

// We deliberately defer chart.js registration to inside the setup callback
// (rather than at module top-level) — top-level side effects in setup files
// have been observed to interfere with Slidev's app/router init chain.
//
// Brand defaults: NTHU purple primary on light-cyan ground, Arial body to
// align with thesis/Group Meeting.pptx master.

export default defineAppSetup(async () => {
  const { Chart, registerables, defaults } = await import('chart.js')
  Chart.register(...registerables)
  defaults.font.family =
    "Arial,'Helvetica Neue',Helvetica,'PingFang TC',sans-serif"
  defaults.font.size = 11
  defaults.color = '#7F1084'           // NTHU purple
  defaults.borderColor = '#E5E0EC'
  ;(defaults as any).animation = false
})
