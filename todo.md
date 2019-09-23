# Reg. Code Organization -  20th Aug
1. [DONE] models.py <- move all model defs here
2. [TODO] make new iterator for evaluation.
    We want to generate all negatives (all s, and o) replaced per positive triple.
    Also need new mechanisms to evaluate (accuracy, recirank).
3. [ALMOST DONE] Convert fb15k and wd147k notebooks to py files.

1. [DONE] Verify the inputs and outputs of MRR, MR, HITS fn
1. Connect argument parsing to change config
1. Write nice data loaders, clean, out of the way...

# ADD MORE TESTS!

## Refactoring 
1. Don't assume a given corruption position. By default it should be None.
    If it is none, you should corrupt `[::2]`th of the data.
    
2. IS_QUINTS is no longer needed. Swapped for STATEMENT_LEN
3. After changing the model, check if embeddings for zero are not changed
    3.1 Saving the model
4. Inflecting relations



## Friday - 20th Sept
Re-wire TransE forward to be more receptive to quints and triples than right now.
