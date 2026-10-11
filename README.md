# \# Database-Backed CPU Scheduler with Analytics

# 

# \## Overview

# This project implements a CPU scheduling engine that integrates with a database backend for persistent task management and analytics. It allows for performance benchmarking across different scheduling algorithms under varying workloads.

# 

# \## Scheduler Engine

# The core scheduler implements three fundamental CPU scheduling policies:

# \* \*\*First-Come, First-Served (FCFS)\*\*

# \* \*\*Shortest Job First (SJF)\*\*

# \* \*\*Priority Scheduling\*\*

# 

# \## Benchmarks \& Scaling

# The project includes concurrency-scaling batch scripts to test the scheduler's performance and database transaction locks under heavy, concurrent task loads.

# 

# \## Workload Simulator

# To train and evaluate the system, the simulation module generates calibrated, synthetic workloads mimicking real-world CPU task distributions.

# 

# \## Analytics \& Decision Tree

# The analytics module uses scikit-learn to build a decision tree model from the simulation data. It predicts the most efficient scheduling algorithm for a given workload profile based on execution metrics like mean wait time and 95th-percentile (p95) turnaround time.

# 

# The trained trees are exported as `tree\_mean.txt` and `tree\_p95.txt`.

