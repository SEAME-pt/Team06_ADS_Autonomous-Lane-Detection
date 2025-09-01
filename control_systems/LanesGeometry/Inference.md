# Inference (TensorRT runtime for lane model)

This document specifies the TensorRT-based inference module used to run the lane network on Jetson Nano, covering scope, interfaces, runtime architecture, error handling, real-time behavior, configuration, V&V, and a concise API reference [10].

## Purpose and scope
The runtime deserializes a prebuilt TensorRT engine, allocates device/host buffers, performs H2D → synchronous execution → D2H per frame, and returns the output tensor as contiguous floats for the downstream lane-processing stage [10]. The design prioritizes deterministic timing and simplicity (single input/output binding, one execution context, synchronous `executeV2`) to fit the control loop budget [10].

## Interfaces and contracts
- Input: preprocessed image tensor flattened as `std::vector<float>` matching the engine’s input binding shape and order; normalization and layout are performed upstream to avoid duplication in the runtime [10].  
- Output: flattened `std::vector<float>` copied from the engine’s output binding; the consumer interprets shape and semantics (e.g., lane mask) consistently with the exported engine [10].  
- Assumptions: exactly one input binding and one output binding; if the engine I/O changes, the binding discovery/allocation must be updated accordingly [10].

## Runtime architecture
- Engine lifecycle: load engine bytes from file → `createInferRuntime` → `deserializeCudaEngine` → `createExecutionContext` (exceptions on failure) [10].  
- Buffering: query `getNbBindings` and `getBindingDimensions` to compute linear sizes; allocate device memory via `cudaMalloc` and host mirrors via `new float[]`; fill the `bindings` array with device pointers, split into input/output vectors for clarity [10].  
- Execution flow (per call): `cudaMemcpy` H2D (input) → `context->executeV2(bindings.data())` (sync) → `cudaPeekAtLastError` + `cudaDeviceSynchronize` → `cudaMemcpy` D2H (output) → return vector [10].

## Error handling and fallbacks
- The runtime throws descriptive exceptions on: missing engine file, runtime/engine/context creation failures, device allocation errors, kernel/runtime errors post-execution, and device sync errors; upstream code should catch and apply a safe fallback (e.g., reuse last valid output or signal degraded mode) [10].  
- Logging: the logger prints warnings/errors to stdout; consider routing to a centralized diagnostic sink in integrated builds [9].

## Real‑time behavior
- Synchronous `executeV2` and explicit synchronization bound the per-frame latency and simplify budgeting with the perception → control chain, avoiding stream coordination overhead on the Nano [10].  
- If end-to-end latency becomes transfer-bound, consider a future variant with `enqueue` + CUDA stream and pinned host memory to overlap H2D/D2H with compute; measure determinism impact before adopting in safety-relevant paths [10].

## Configuration and compatibility
- Parameters: `engine_path` (filesystem), expected input dtype/shape (as exported), and buffer sizes inferred at runtime from bindings [10].  
- Compatibility: one engine/runtime/context per module instance; if multi-threading is introduced, prefer one execution context per thread to avoid contention; confirm versions jointly with the platform build to ensure reproducibility [10].

## Verification and validation (V&V)
- Unit tests: engine load/destroy cycles (no leaks), binding size checks, H2D/D2H integrity on known patterns, and exception paths for simulated allocation/execute failures [10].  
- Integration tests: end-to-end inference timing (avg/worst/jitter) under load; regression of output tensor distribution on a fixed dataset across engine and runtime versions, archived with hashes for traceability [10].

## Known limitations and studied options
- No support (current build) for multiple bindings, dynamic shapes, or multiple optimization profiles; extend binding discovery and context setup if needed by future engines [10].  
- No CUDA streams/overlap path; kept synchronous for determinism; an overlapped variant can be introduced behind a compile-time/runtime option after timing validation on target hardware [10].

## API reference (concise)
- `TensorRTInference(const std::string& engine_path)`: loads and deserializes the engine, creates an execution context, and allocates device/host buffers for all bindings (throws on failure) [10].  
- `std::vector<float> infer(const std::vector<float>& inputData)`: copies input to device, executes the network synchronously, checks CUDA errors, synchronizes, copies output to host, and returns it (throws on failure) [10].  
- `~TensorRTInference()`: frees device buffers and host mirrors, destroys context/engine/runtime (no-throw cleanup) [10].
