-- Final selection over the reviewed wide pipeline snapshot.
SELECT
  M,
  gpu0_sequential_ms,
  gpu0_overlap_ms,
  gpu0_speedup,
  gpu1_sequential_ms,
  gpu1_overlap_ms,
  gpu1_speedup
FROM pipeline_curve
ORDER BY M;
