(() => {
  const method = document.getElementById('threshold_method');
  const setting = document.getElementById('threshold_setting');
  const label = document.getElementById('threshold-setting-label');
  const help = document.getElementById('threshold-setting-help');

  function updateThresholdFields() {
    if (!method || !setting || !label) return;
    const isPercentile = method.value === 'percentile';
    label.textContent = isPercentile ? 'Percentile (90–99.9)' : 'Standard deviation multiplier (1–5)';
    setting.min = isPercentile ? '90' : '1';
    setting.max = isPercentile ? '99.9' : '5';
    setting.step = isPercentile ? '0.1' : '0.5';
    const current = Number(setting.value);
    if (isPercentile && (current < 90 || current > 99.9)) setting.value = '99.0';
    if (!isPercentile && (current < 1 || current > 5)) setting.value = '3.0';
    if (help) help.textContent = isPercentile
      ? 'Higher percentiles generally create a more conservative threshold.'
      : 'The threshold is the training-error mean plus this many standard deviations.';
  }
  if (method) {
    method.addEventListener('change', updateThresholdFields);
    updateThresholdFields();
  }

  document.querySelectorAll('form[data-busy-text]').forEach((form) => {
    form.addEventListener('submit', () => {
      const submit = form.querySelector('button[type="submit"]');
      const busy = form.querySelector('.busy-indicator');
      if (submit) {
        submit.disabled = true;
        const span = submit.querySelector('span');
        if (span) span.textContent = form.dataset.busyText || 'Working…';
      }
      if (busy) busy.hidden = false;
    });
  });
})();
