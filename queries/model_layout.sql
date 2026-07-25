-- Final selection over the reviewed Llama-3.2-1B BF16 component layout.
SELECT
  component,
  bytes,
  MiB
FROM model_layout
ORDER BY bytes DESC;
