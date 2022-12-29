## Version 1
- Baseline
- Performance evaluation:
  - Avg execution time: ... ms (... runs)

## Version 2 (baseline: ver 1)
- `seamCarvingGpu()` now uses ptr swapping trick on 2 pointers instead of operating on just 1 array
- `carveSeamKernel()` has also been updated to reflect that change and can now be executed on more than 1 block as well as on a 2D grid
- Performance evaluation:
  - Avg execution time: ... ms (... runs)
  - Significant reduction in execution time