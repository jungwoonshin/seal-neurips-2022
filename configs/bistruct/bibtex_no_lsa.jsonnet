// BiStruct ablation: no LSA refinement (beta=0)
// Tests whether alignment alone captures dependencies
local base = import 'bibtex.jsonnet';

base {
    "model"+: {
        "beta": 0.0,  // disable refinement loss (LSA gets no gradient signal)
    }
}
