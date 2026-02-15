Auto Quantum Machine Learning for Multisource Classification

# Tips

- if using OSX, be aware that `pennylane` hates macs. You'll need to use `cpu` for computations with torch. Possibly
even modify some of the libraries or environmental variables, just so that some internal module doesn't use it.
