# Database-Backed-CPU-Scheduler-with-Analytics

A CPU scheduling simulator that implements multiple scheduling algorithms and stores scheduling events in a relational database for SQL-based analysis and comparison.

## Overview

The project combines Operating Systems and DBMS concepts by simulating CPU scheduling, logging process events, and analyzing the results using SQL.

## Features

- Multiple CPU scheduling algorithms
- Process scheduling simulation
- Scheduling event logging
- Relational database storage
- SQL-based performance analysis
- Waiting time, turnaround time, response time, and CPU utilization analysis
- Context switch simulation
- Database transactions and indexing

## OS + DBMS Integration

The scheduler generates scheduling events, which are handled by a separate logging component and stored in the database. SQL is then used to analyze and compare scheduling performance.

## Technologies

- C / C++ / Python
- MySQL / PostgreSQL / SQLite
- SQL

## Future Scope

- Scheduling visualization
- Real process monitoring
- Linux kernel-level scheduling experiments
- Advanced SQL analytics

## License

This project is developed for educational purposes.
