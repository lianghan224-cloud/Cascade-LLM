-- Final selection over CUDA Driver API medians reconciled from both raw files.
SELECT
  size_bytes,
  size,
  gpu0_event_us,
  gpu0_GBps,
  gpu1_event_us,
  gpu1_GBps
FROM h2d_chunks
ORDER BY size_bytes;
