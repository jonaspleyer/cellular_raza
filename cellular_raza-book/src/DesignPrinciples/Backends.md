# Backend
## `cpu_os_threads`
- Intrinsic parallelization over physical simulation domain
- Strongly prefer static dispatch (frequent use of generics, avoid `Box<dyn Trait>`)
- Consider memory-locality, efficient cache-usage and thread-to-thread latency
- reduce heap allocations

## `cpu_rayon`
