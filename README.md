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
Standard xv6 does not save Floating Point Unit (FPU) states during context switches to keep the scheduler simple. **SeeXv6** enables **eager FPU/SSE state saving**, ensuring that high-precision weights and biases are preserved across process threads during training.

---

## 📂 Branch Structure
* **`xv6-main`**: The original, unmodified MIT xv6 source. Used as a clean reference for stability.
* **`ml-dev`**: The active development branch for ML-specific features, drivers, and syscalls.

---

## 📜 Credits & Licensing

**SeeXv6** is a derivative work of **xv6-riscv** (or xv6-x86), developed by the **MIT Parallel and Distributed Operating Systems Group (PDOS)**. 

I am deeply grateful to the original authors for providing a clean, pedagogical foundation for operating systems research. The original source code is available at the [MIT xv6 GitHub](https://github.com/mit-pdos/xv6-riscv).

* **Original xv6 Authors:** Russ Cox, Cliff Frey, Xiao Yu, Frans Kaashoek, Nickolai Zeldovich, and Robert Morris.
* **SeeXv6 Derivative:** Developed by Shuo Zheng for ML-optimized kernel research.
