# section: reduction_critic_escape

THIS CARD IS AN ESCAPE CARD. Its parent was reduced to a known family at S2, and this card claims, in non_reduction_argument, to break the property that reduction relied on. Below are the argument and the parent's reduction.

Test the argument specifically, in addition to your usual work:

1. Check that broken_invariant is true of the card's own equations: find the update or readout step it names and confirm that step, as written, violates the invariant. An argument about a step that is not in the core, or that the equations do not support, does not count.
2. Try to repair the parent's mapping. Start from the strongest repair you can find: a change of basis, a lifted or larger state (including one that grows with n or stores every token), a learned or fixed feature map applied to keys, values or queries, data-dependent gates or decays, a second memory of the same family, or the known family run over a transformed token stream. Also try the families next to the parent's (for a delta rule: linear attention, RLS, SSM, Hopfield; for a data structure: its textbook variants). If any repair maps this card onto a known form, report that reduction with its mapping, as you would for any card, and declare in `comparators` the parent's mapping as published and the strongest repair, with every transformation typed from the REPAIR POLICY (`other` for a change the policy does not list).
3. Judge why_unrepairable: does the card name the strongest repair, and is the reason that repair fails correct?

Say in why whether the argument held, and which repair you tried that came closest. When the argument fails, the card reduces like any other card; do not give credit for the argument itself.
