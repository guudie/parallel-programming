## Version 1
- Baseline

## Version 2
- `seamCarvingGpu()` now uses ptr swapping trick on 2 pointers instead of operating on just 1 array
- `carveSeamKernel()` has also been updated to reflect that change and can now be executed on more than 1 block as well as on a 2D grid