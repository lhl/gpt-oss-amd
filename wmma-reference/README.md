WMMA Reference Folder

What’s here
- `rdna35_instruction_set_architecture.md`: RDNA3(.5) ISA notes relevant to WMMA.
- `rocWMMA/`: Official rocWMMA library (submodule).
- `rocm_wmma_samples/`: Example HIP kernels using WMMA (submodule).
- `flash-attention-v2-RDNA3-minimal/`: Minimal FA2 RDNA3 example (submodule).
- `amd_matrix_instruction_calculator/`: AMD matrix instruction calculator (submodule).

Get the submodules
- Initialize/update the references:
  - `git submodule update --init --recursive`

Tips
- Cross-reference rocWMMA kernels and the ISA notes when mapping MFMA→WMMA.
- Many large assets are covered by Git LFS via the repo-level `.gitattributes`.

