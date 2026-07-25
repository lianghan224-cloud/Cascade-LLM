-- Final selection over exact-shape GPU0 projection measurements.
SELECT
  M,
  elapsed_ms,
  effective_TFLOPs,
  nominal_weight_GBps
FROM compute_gpu0
ORDER BY M;
