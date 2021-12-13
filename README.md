# DFHist 

**Utilities for histories and caches of pandas dataframes**

J. M. F. Tsang (j.m.f.tsang@cantab.net)

---

## What is this?

DFHist lets you cache the results of functions that produce pandas dataframes,
as well as maintaining a history of the dataframes on disk (as CSV files).

This is useful for running functions that produce large dataframes that can
change over time, such as running SQL queries using `pd.read_sql_query` or 
querying an API.
