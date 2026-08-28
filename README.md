Auto Quantum Machine Learning for Multisource Classification

# Tips

- if using OSX, be aware that `pennylane` hates macs. You'll need to use `cpu` for computations with torch. Possibly
even modify some of the libraries or environmental variables, just so that some internal module doesn't use it.

# TODOS

**SD**:
- Take care of naming the classes, modules and variables properly. Examples include:
  - Distinctive names for devices in hybrid models using torch. "`device`" is commonly used by both `pennylane` and `torch`
- Typehint the module! I had to look for the information about the intended type of the variables in more places than
in should need to!
- Parallelizing
