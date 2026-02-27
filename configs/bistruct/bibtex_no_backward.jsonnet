// BiStruct ablation: no backward path (alpha=0)
// Tests whether LSA alone captures dependencies
local base = import 'bibtex.jsonnet';

base {
    "model"+: {
        "alpha": 0.0,  // disable alignment loss (backward path has no gradient signal)
    }
}
