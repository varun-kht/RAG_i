# Python 3.11 Feature Guide & Overview

Python 3.11 was officially released on October 24, 2022. It brings major speedups, enhanced error tracebacks, and new language features.

## Key Features in Python 3.11

### 1. Faster CPython Project
Python 3.11 is between **10% to 60% faster** than Python 3.10, with an average speedup of 25% on standard benchmarks. Key optimizations include:
- Specialized adaptive bytecode interpreter.
- Inlined function calls for lower call overhead.
- Faster object creation and attribute access.

### 2. Fine-Grained Error Tracebacks
Error messages in Python 3.11 now pinpoint the exact expression that caused the error, rather than just pointing to the line number. This is extremely helpful when debugging nested dict lookups or binary operations.

### 3. Exception Groups (`ExceptionGroup` and `except*`)
Introduces the ability to raise and handle multiple unrelated exceptions simultaneously using `ExceptionGroup` and the `except*` syntax. Useful in asynchronous programming (`asyncio`).

### 4. `Self` Type Annotation
The `typing.Self` annotation allows methods that return an instance of their class to be annotated cleanly without referencing forward declared class names.

### 5. TOML Support in Standard Library (`tomllib`)
Python 3.11 includes `tomllib` in the standard library for parsing TOML files natively without third-party dependencies.
