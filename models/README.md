# models/

ML interatomic potential weights (ONNX format, e.g. ANI-2x / MACE / Allegro exports
consumed by `onnxruntime-web` on the WebGPU backend) live in this directory.

These files are deliberately **not** covered by a blanket `.gitignore` rule -- see
the root `.gitignore` note. Large `*.onnx` artifacts should be committed here
consciously (or, once the repo grows, moved to Git LFS or attached as GitHub
release assets -- that decision is out of scope for the initial scaffold).

Naming convention (proposed, adjust as real models land):

```
models/
  <family>-<variant>.onnx        e.g. ani2x-b973c.onnx
  <family>-<variant>.json        sidecar metadata: element order, cutoff radius,
                                  training set citation, output units
```

Nothing is populated here yet. The `spike/ani2x-onnx/` directory (owned by a
different agent) is exploring ONNX export of a real ML potential; once that
lands it may be promoted into this directory.
