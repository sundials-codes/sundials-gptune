from csv import DictWriter
from unicodedata import decomposition
import networkx as nx
import matplotlib.pyplot as plt
import gptune as gpt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

class DecisionTreeNode:
  def __init__(self, label, tuning_params=[], constraints={}):
    self.label = label
    self.tuning_params = tuning_params
    self.constraints = constraints

  def __repr__(self):
    cls = self.__class__.__name__
    return f'{cls}(label={repr(self.label)})'

  def __hash__(self):
    return hash(repr(self))

class NoParamNode(DecisionTreeNode):
  def __init__(self, label):
    super().__init__(label=label)

  def __repr__(self):
    cls = self.__class__.__name__
    return f'{cls}(label={repr(self.label)})'

  def __hash__(self):
    return hash(repr(self))

class DecisionTree:

  def __init__(self):
    self.choose_implicit_or_explicit = DecisionTreeNode(label='implicit or explicit?', tuning_params=[
      gpt.Categoricalnorm(['implicit', 'explicit'], transform='onehot', name='implicit or explicit?')
    ])

    self.implicit = DecisionTreeNode(label='implicit methods', tuning_params=[
      gpt.Categoricalnorm(['BDF', 'Adams'], transform='onehot', name='LMM'),
      # TODO(CJB): most of these are CVODE specific
      gpt.Integer(1, 5, transform="normalize", name="maxord"),
      gpt.Real(1e-5, 0.9, transform="normalize", name="nonlin_conv_coef"),
      gpt.Integer(3, 50, transform="normalize", name="max_conv_fails"),
      gpt.Real(1e-2, 0.9, transform="normalize", name="eta_cf"),
      gpt.Real(1, 5, transform="normalize", name="eta_max_fx"),
      gpt.Real(0, 0.9, transform="normalize", name="eta_min_fx"),
      gpt.Real(1.1, 20, transform="normalize", name="eta_max_gs"),
      gpt.Real(1e-2, 1, transform="normalize", name="eta_min"),
      gpt.Real(1e-2, 0.9, transform="normalize", name="eta_min_ef")
    ], constraints={'cst1': 'eta_max_fx > eta_min_fx'})

    self.choose_nonlinear_solver = DecisionTreeNode(label='nonlinear solver', tuning_params=[
      gpt.Categoricalnorm(['Newton', 'Fixedpoint'], transform='onehot', name='NLS')
    ])

    self.newton = DecisionTreeNode(label='newton', tuning_params=[
      gpt.Real(1e-5, 0.9, transform="normalize", name="epslin"),
    ])

    self.fixed_point = DecisionTreeNode(label='fixed-point', tuning_params=[
      gpt.Integer(1, 20, transform="normalize", name="fixedpointvecs")
    ])

    self.choose_matrix_based_or_free = NoParamNode(label='matrix-based or matrix-free?')

    self.choose_direct_or_iterative = DecisionTreeNode(label='direct or iterative?', tuning_params=[
      gpt.Integer(3, 500, transform="normalize", name="maxl"),
    ])

    self.linear_solver_matrix_free = DecisionTreeNode(label='matrix free', tuning_params=[
      gpt.Categoricalnorm(['gmres', 'bicgstab'])
    ])

    self.linear_solver_matrix_based_direct = DecisionTreeNode(label='direct', tuning_params=[
      gpt.Categoricalnorm(['magma_batched_lu']),
      gpt.Integer(1, 200, transform="normalize", name="msbp"),
      gpt.Integer(1, 200, transform="normalize", name="msbj"),
      gpt.Real(1e-2, 0.5, transform="normalize", name="dgmax"),
    ], constraints={"msbpmsbj": "msbj >= msbp"})

    self.linear_solver_matrix_based_iterative = DecisionTreeNode(label='iterative', tuning_params=[
      gpt.Categoricalnorm(['ginkgo_gmres', 'ginkgo_bicgstab'])
    ])

    self.explicit = DecisionTreeNode(label='explicit', tuning_params=[
      gpt.Categoricalnorm(['HEUN_EULER_2_1_2', 'DORMAND_PRINCE_7_4_5'])
    ])

  def graph(self):
    decision_tree = nx.DiGraph()
    decision_tree.add_node(self.choose_implicit_or_explicit)
    decision_tree.add_node(self.implicit)
    decision_tree.add_node(self.explicit)
    decision_tree.add_node(self.choose_nonlinear_solver)
    decision_tree.add_node(self.choose_matrix_based_or_free)
    decision_tree.add_node(self.choose_direct_or_iterative)
    decision_tree.add_node(self.linear_solver_matrix_free)
    decision_tree.add_node(self.linear_solver_matrix_based_direct)
    decision_tree.add_node(self.linear_solver_matrix_based_iterative)
    decision_tree.add_edge(self.choose_implicit_or_explicit, self.implicit)
    decision_tree.add_edge(self.choose_implicit_or_explicit, self.explicit)
    decision_tree.add_edge(self.implicit, self.choose_nonlinear_solver)
    decision_tree.add_edge(self.choose_nonlinear_solver, self.newton)
    decision_tree.add_edge(self.choose_nonlinear_solver, self.fixed_point)
    decision_tree.add_edge(self.newton, self.choose_matrix_based_or_free)
    decision_tree.add_edge(self.choose_matrix_based_or_free, self.choose_direct_or_iterative)
    decision_tree.add_edge(self.choose_matrix_based_or_free, self.linear_solver_matrix_free)
    decision_tree.add_edge(self.choose_direct_or_iterative, self.linear_solver_matrix_based_iterative)
    decision_tree.add_edge(self.choose_direct_or_iterative, self.linear_solver_matrix_based_direct)
    return decision_tree

def path_params(G, path_nodes):
  w = []
  for ind, nd in enumerate(path_nodes[1:]):
      prev = path_nodes[ind]
      w += nd.tuning_params
  return w

def path_constraints(G, path_nodes):
  w = {}
  for ind, nd in enumerate(path_nodes[1:]):
      prev = path_nodes[ind]
      w = dict(w, **nd.constraints)
  return w

def plot_full_tree(G, save_fig=True, save_path='decision_tree.png'):
  decision_tree = full_tree()
  pos = graphviz_layout(decision_tree, prog="dot")
  nx.draw(decision_tree, pos)
  nx.draw_networkx_labels(decision_tree, pos, labels={node:node.label for node in decision_tree})
  nx.draw_networkx_edge_labels(decision_tree, pos, labels={edge:edge.tuning_params for edge in decision_tree})
  if save_fig:
    plt.savefig(save_path)

# path = [choose_implicit_or_explicit, implicit, choose_nonlinear_solver, newton, choose_matrix_based_or_free, choose_direct_or_iterative, linear_solver_matrix_based_direct]

# print(path_params(decision_tree, path))
# print(path_constraints(decision_tree, path))

# pos = graphviz_layout(decision_tree, prog="dot")
# nx.draw(decision_tree, pos)
# nx.draw_networkx_labels(decision_tree, pos, labels={node:node.label for node in decision_tree})
# nx.draw_networkx_edge_labels(decision_tree, pos, labels={edge:edge.tuning_params for edge in decision_tree})
# plt.savefig('decision_tree.png')
