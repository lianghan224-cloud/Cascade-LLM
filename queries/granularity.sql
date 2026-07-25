-- Final selection over the 24 GB/s H2D-window calculation.
SELECT
  M,
  compute_ms,
  hideable_MiB_at_24_GBps,
  recommendation
FROM granularity
ORDER BY M;
