import jax
import ipdb

class pdb():
    def set_trace():
        if jax.process_index() == 0:
            ipdb.set_trace()