Scope: repository-wide

Purpose
- This repo is porting AMD GPU code from CDNA (MFMA) to RDNA (WMMA).
- For WMMA implementation details and examples, use the `wmma-reference/` folder.

How to Work in This Repo
- Start with `readme.md` and `WORKING.md` for project context and current status.
- Use `wmma-reference/` for RDNA/WMMA specs and reference implementations.
- Submodules live under `wmma-reference/`. If they appear empty, run:
  - `git submodule update --init --recursive`

Reference Contents (`wmma-reference/`)
- `rdna35_instruction_set_architecture.md`: RDNA3(.5) ISA notes for WMMA.
- `rocWMMA/`: Official rocWMMA library reference.
- `rocm_wmma_samples/`: Small WMMA sample kernels.
- `flash-attention-v2-RDNA3-minimal/`: Minimal FA2 RDNA3 example.
- `amd_matrix_instruction_calculator/`: AMD matrix instruction calculator.

Notes
- Large binaries (models, images, archives) are tracked via Git LFS per `.gitattributes`.
- Keep changes focused; follow existing code style and structure.

