-- Final selection over the reviewed pipeline_chart snapshot materialized by
-- benchmarks/build_report_artifact.py from both saved GPU result files.
SELECT
  M,
  M_label,
  gpu,
  speedup,
  sequential_ms,
  overlap_ms
FROM pipeline_chart
ORDER BY M, gpu;
