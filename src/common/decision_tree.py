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
      # gpt.Categoricalnorm(['implicit', 'explicit'], transform='onehot', name='implicit or explicit?')
    ])

    self.implicit = DecisionTreeNode(label='implicit methods', tuning_params=[
      gpt.Categoricalnorm(['BDF', 'Adams'], transform='onehot', name='LMM'),
      # TODO(CJB): most of these are CVODE specific
      gpt.Real(0.1, 0.9, transform="normalize", name="nonlin_conv_coef"),
      gpt.Integer(3, 50, transform="normalize", name="max_conv_fails"),
      gpt.Real(1e-2, 0.9, transform="normalize", name="eta_cf"),
      gpt.Real(1, 5, transform="normalize", name="eta_max_fx"),
      gpt.Real(0.0, 0.9, transform="normalize", name="eta_min_fx"),
      gpt.Real(1.1, 20, transform="normalize", name="eta_max_gs"),
      gpt.Real(1e-2, 1, transform="normalize", name="eta_min"),
      gpt.Real(1e-2, 0.9, transform="normalize", name="eta_min_ef")
    ], constraints={'eta_max_min_fx': 'eta_max_fx > eta_min_fx'})

    self.implicit_bdf = DecisionTreeNode(label='BDF', tuning_params=[
      gpt.Integer(1, 5, transform="normalize", name="max_order"),
    ], constraints={'lmm_is_bdf': 'LMM == \'BDF\''})

    self.implicit_adams = DecisionTreeNode(label='Adams', tuning_params=[
      gpt.Integer(1, 12, transform="normalize", name="max_order"),
    ], constraints={'lmm_is_adams': 'LMM == \'Adams\''})

    self.choose_nonlinear_solver = DecisionTreeNode(label='nonlinear solver', tuning_params=[
      gpt.Categoricalnorm(['Newton', 'fixed_point'], transform='onehot', name='nonlinear_solver')
    ])

    self.newton = DecisionTreeNode(label='Newton', tuning_params=[
      gpt.Real(0.1, 0.9, transform="normalize", name="epslin"),
    ])

    self.fixed_point = DecisionTreeNode(label='fixed_point', tuning_params=[
      gpt.Integer(1, 20, transform="normalize", name="max_fp_accel")
    ])

    self.choose_matrix_based_or_free = NoParamNode(label='matrix-based or matrix-free?')

    self.matrix_based = DecisionTreeNode(label='matrix_based', tuning_params=[
      gpt.Integer(1, 200, transform="normalize", name="msbp"),
      gpt.Integer(1, 200, transform="normalize", name="msbj"),
      gpt.Real(0.1, 0.5, transform="normalize", name="dgmax"),
    ], constraints={"msbpmsbj": "msbj >= msbp"})

    self.choose_direct_or_iterative = NoParamNode(label='direct or iterative?')

    self.matrix_based_direct = DecisionTreeNode(label='direct', tuning_params=[
      gpt.Categoricalnorm(['magma_batched_lu'], transform="onehot", name="linear_solver"),
    ])

    self.matrix_based_iterative = DecisionTreeNode(label='iterative', tuning_params=[
      gpt.Categoricalnorm(['ginkgo_GMRES', 'ginkgo_BICGSTAB'], transform="onehot", name="linear_solver")
    ])

    self.preconditioner = DecisionTreeNode(label='preconditioner', tuning_params=[
      # TODO(CJB): add preconditioner params - will need to think about how this works with general SUNDIALS
    ])

    self.matrix_free = DecisionTreeNode(label='matrix_free', tuning_params=[
      gpt.Categoricalnorm(['GMRES', 'BCGS'], transform="onehot", name="linear_solver"),
      # TODO(CJB): need a sensible way to modify ranges for parameters from tuning driver since an appropriate range is not the same for all problems
      # gpt.Integer(3, 500, transform="normalize", name="maxl")
      gpt.Integer(1, 500, transform="normalize", name="maxl") # special for PELE
    ])

    self.explicit = DecisionTreeNode(label='explicit', tuning_params=[
      gpt.Categoricalnorm([
        'ARKODE_HEUN_EULER_2_1_2',
        'ARKODE_BOGACKI_SHAMPINE_4_2_3',
        'ARKODE_ARK324L2SA_ERK_4_2_3',
        'ARKODE_ZONNEVELD_5_3_4',
        'ARKODE_ARK436L2SA_ERK_6_3_4',
        'ARKODE_SAYFY_ABURUB_6_3_4',
        'ARKODE_CASH_KARP_6_4_5',
        'ARKODE_FEHLBERG_6_4_5',
        'ARKODE_DORMAND_PRINCE_7_4_5',
        'ARKODE_ARK548L2SA_ERK_8_4_5',
        'ARKODE_VERNER_8_5_6',
        'ARKODE_FEHLBERG_13_7_8',
        'ARKODE_KNOTH_WOLKE_3_3',
        'ARKODE_ARK437L2SA_ERK_7_3_4',
        'ARKODE_ARK548L2SAb_ERK_8_4_5'
      ], transform="onehot", name="erk_method")
    ])



    self.dt_graph = nx.DiGraph()
    self.dt_graph.add_node(self.choose_implicit_or_explicit)
    self.dt_graph.add_node(self.implicit)
    self.dt_graph.add_node(self.explicit)
    self.dt_graph.add_node(self.choose_nonlinear_solver)
    self.dt_graph.add_node(self.choose_matrix_based_or_free)
    self.dt_graph.add_node(self.choose_direct_or_iterative)
    self.dt_graph.add_node(self.matrix_free)
    self.dt_graph.add_node(self.matrix_based)
    self.dt_graph.add_node(self.matrix_based_direct)
    self.dt_graph.add_node(self.matrix_based_iterative)
    self.dt_graph.add_edge(self.choose_implicit_or_explicit, self.implicit)
    self.dt_graph.add_edge(self.choose_implicit_or_explicit, self.explicit)
    self.dt_graph.add_edge(self.implicit, self.implicit_bdf)
    self.dt_graph.add_edge(self.implicit, self.implicit_adams)
    self.dt_graph.add_edge(self.implicit_bdf, self.choose_nonlinear_solver)
    self.dt_graph.add_edge(self.implicit_adams, self.choose_nonlinear_solver)
    self.dt_graph.add_edge(self.choose_nonlinear_solver, self.newton)
    self.dt_graph.add_edge(self.choose_nonlinear_solver, self.fixed_point)
    self.dt_graph.add_edge(self.newton, self.choose_matrix_based_or_free)
    self.dt_graph.add_edge(self.choose_matrix_based_or_free, self.matrix_based)
    self.dt_graph.add_edge(self.matrix_based, self.choose_direct_or_iterative)
    self.dt_graph.add_edge(self.choose_direct_or_iterative, self.matrix_based_iterative)
    self.dt_graph.add_edge(self.choose_direct_or_iterative, self.matrix_based_direct)
    self.dt_graph.add_edge(self.choose_matrix_based_or_free, self.matrix_free)

  def graph(self):
    return self.dt_graph

  def expand_path(self, list_path):
    expanded_path = []
    for ind, nd in enumerate(list_path):
      if nd == '*':
        for nd in nx.bfs_successors(self.dt_graph, getattr(self, list_path[ind-1])):
          if ind+1 < len(list_path):
            if nd != list_path[ind+1]:
              expanded_path.extend(nd[1])
          else:
            expanded_path.extend(nd[1])
      else:
        expanded_path.append(getattr(self, nd))
    return expanded_path

  def draw(self, save_fig=True, save_path='decision_tree.png'):
    G = self.graph()
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos, labels={node:node.label for node in G})
    # nx.draw_networkx_edge_labels(G, pos, labels={edge:edge.tuning_params for edge in G})
    if save_fig:
      plt.savefig(save_path)

def path_params(G, path_nodes):
  w = []
  for nd in path_nodes[1:]:
      w += nd.tuning_params
  return w

def path_constraints(G, path_nodes):
  w = {}
  for nd in path_nodes[1:]:
      w = dict(w, **nd.constraints)
  return w



