# SeeXv6: A Deep Learning Optimized Operator

**SeeXv6** is a research-oriented operating system built on the foundation of the MIT **xv6** kernel. While the standard xv6 is designed for pedagogical simplicity, this fork is being re-engineered to support high-throughput **Deep Neural Network (DNN)** training workloads at the kernel level.

---

## 🚀 The Vision
Most modern ML stacks sit atop general-purpose kernels (Linux/Unix) that prioritize "fair" CPU scheduling for general tasks. **SeeXv6** flips the script: the kernel treats **Tensors** as first-class citizens and the **Accelerator (GPU/TPU)** as the primary processor.

## 🛠 Core Architectural Goals

### 1. Kernel-Level Math Primitives
Instead of relying solely on user-space libraries (like NumPy), we are implementing **System Calls for Linear Algebra**. This allows the OS to manage hardware registers directly for matrix multiplications, significantly reducing the overhead of context switching between the application and the hardware.

### 2. High-Performance Memory (Large Pages)
Standard 4KB memory pages cause massive Translation Lookaside Buffer (TLB) misses during large DNN training. We are implementing **Huge Pages (2MB)** and a **Zero-Copy memory architecture** to allow seamless data transfer between the CPU and ML accelerators without redundant data movement.

### 3. FPU-Aware Scheduling
Standard xv6 does not save Floating Point Unit (FPU) states during context switches to keep the scheduler simple. **ML-xv6** enables **eager FPU/SSE state saving**, ensuring that high-precision weights and biases are preserved across process threads during training.

---

## 📂 Branch Structure
* **`xv6-main`**: The original, unmodified MIT xv6 source. Used as a clean reference for stability.
* **`ml-dev`**: The active development branch for ML-specific features, drivers, and syscalls.

## 🚧 Current Roadmap
- [ ] **Phase 1: FPU Integration** - Enable x87 FPU / SSE support in the kernel (Crucial for decimal math).
- [ ] **Phase 2: ML System Calls** - Implement the `sys_matmul` system call for kernel-level matrix operations.
- [ ] **Phase 3: Tensor Memory Management** - Create a contiguous memory allocator for large tensor buffers.
- [ ] **Phase 4: Hardware Interface** - Basic PCIe driver support to communicate with external hardware.
