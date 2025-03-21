Fall 2024 Tasks!
================

This is the To-Do list for Fall 2024, outlining both must-have and nice-to-have tasks for the GRiD project.

Must Haves
-----------

1. **Base Case and Code Integration**

-  [ ] Get everything working on all base cases (Expected 10/25)
-  [ ] Integrate all code from the academic year into mainline A2R/GRiD
-  [ ] Continue adding documentation for each function (Google Test, Pybind)
-  [ ] Add Github badges for tests passing and documentation coverage (*See example: https://github.com/rl-tools/rl-tools*)
-  [ ] Set up standard testing suite (may not be straightforward)
-  [ ] Test CUDA vs. Python automated testing to confirm correct values
-  [ ] Add Python wrappers for calling CUDA
-  [ ] GPU autointegration with testing? Docker?
-  [ ] Ensure Python working bindings in the bindings directory

2. **Fixed Base Tasks**

-  [ ] Merge Danelle's CRBA and ABA on the GPU
-  [ ] Finish CRBA implementation
-  [ ] Merge with other new code

3. **Floating Base Tasks**

-  [ ] 2nd Order Dynamics Gradients (IDSVA) in Python for floating base and CUDA on the GPU
-  [ ] Complete ABA, CRBA implementations on GPU
-  [ ] Test and provide documentation for the usage of external forces
-  [ ] Merge PR for RBDReference algorithms
-  [ ] Test algorithms on Atlas and fix memory issues (if any)
-  [ ] Integrate with Trajopt
-  [ ] DDP GPU vs. CPU comparison:
-  [ ] Address memory issues and debug mode
-  [ ] Handle memory challenges for large URDFs (Does Atlas work with GRiD codegen?)
-  [ ] Optimize memory (e.g., F matrix in `minv`)
-  [ ] Build automated testing for GRiD with Docker and CI

4. **Software and Documentation Tasks**

-  [ ] Begin creating GitHub issues and a massive to-do list (ask team members to add anything)
-  [ ] Software stack for quadruped with demo
-  [ ] Stabilize baseline process
-  [ ] Automate building, testing, and documenting for GRiD
-  [ ] Follow Harvard contribution guidelines and add workflow to auto-add acknowledgements to people who commit/PR to repo. 

Nice to Haves
-------------

1. **Extended Work**

-  [ ] Consider extending work into a paper for conferences (e.g., 2nd order DDP | Parallel DDP paper)
-  [ ] Potential paper on supporting contact

2. **CUDA Code Optimization**

-  [ ] Develop a framework for writing portable, efficient CUDA code
-  [ ] Interactive Jupyter notebooks for demonstrations of code functionality with Python bindings

3. **Docker and Automation**

-  [ ] Build a fully self-contained Docker environment for reproducible builds
-  [ ] Add automated benchmarking and profiling pipeline for CUDA and Python performance
-  [ ] Integrate profiling tools like NVIDIA Nsight or `nvprof` for CUDA kernel performance

4. **Collaborations and Open Source**

-  [ ] Work with other GRiD teams (MCGPU PDDP) to help leverage GRiD
-  [ ] Set up open source contribution guidelines
-  [ ] Track the number of downloads
-  [ ] Consider incorporating the Glass library as a submodule
-  [ ] Add GRiDBenchmarks as a GRiD submodule

Goals
-----

-  [ ] Launch GRiD according to open-source guidelines with unit testing and benchmarking
-  [ ] Full online documentation for algorithms, including interactive demos
-  [ ] Create optimization pipeline and framework for generating efficient, scalable code for GPUs
-  [ ] Get well-documented testing demos on real hardware (videos and interactive notebooks promoting GRiD/A2R lab packages)

Week-by-Week Timeline
----------------------

**By October 11th**

-  [ ] Pull requests for floating base → main branch compatibility
-  [ ] Begin setting up automated testing suite for Python vs. CUDA
-  [ ] Obtain hardcoded value tests and results for all RBDReference algorithms
-  [ ] Ensure testing with external forces usage
-  [ ] Merge GRiDCodeGen code and update Readmes for submodules
-  [ ] Add documentation for code and algorithm testing instructions

**By November 15th**

-  [ ] Complete optimization pipeline for code generation, memory management, and kernel fusion
-  [ ] Plan for full GRiD launch
-  [ ] Coordinate paper drafts and potential conference submissions
-  [ ] Collaborate with the GRiD team to get documentation up and running (Sphinx or text files)
-  [ ] Begin setting up open source contribution guidelines
-  [ ] Finish setting up automated testing suite (Python vs. CUDA for all URDFs)
-  [ ] Implement automatic testing for Python vs. CUDA with Pass/Fail status
-  [ ] Add GitHub badge for testing: "Your commit has been tested successfully!"
-  [ ] Finish floating base ABA, CRBA implementations on GPU and have everything tested
-  [ ] Merge all GRiD forks/branches into the main branch

**By November 29th**

-  [ ] Complete optimization pipeline with auto-tuning, memory management, and dynamic scheduling
-  [ ] Have Sphinx/documentation up-to-date with unit testing framework (Catch2 or Google Test)
-  [ ] Begin testing on real hardware (quadruped software stack)
-  [ ] Finalize GRiD's open-source code guidelines and operational status

