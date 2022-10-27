from SundialsDecisionTree import *

decision_tree = DecisionTree()
decision_tree.draw()
decision_tree_path = decision_tree.expand_path(
    [
        "choose_implicit_or_explicit",
        "implicit",
        "implicit_bdf",
        "choose_nonlinear_solver",
        "fixed_point",
        "*",
    ]
)
print(decision_tree_path)
