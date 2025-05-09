import os
import json
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras import backend as K
import seaborn as sns
from tensorflow.keras.callbacks import Callback, EarlyStopping
from rdkit.Chem import Descriptors
from keras import Model, callbacks
from contextlib import redirect_stdout
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import optuna
from optuna.integration import TensorBoardCallback
from optuna.pruners import MedianPruner
import networkx as nx
import csv

import subprocess
def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'], 
                            stdout=subprocess.PIPE, text=True)
    total_memory = int(result.stdout.strip())
    print(f"GPU 总内存: {total_memory} MB")
    return total_memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # 获取 GPU 的总内存（以 MB 为单位）
            total_memory_mb = get_gpu_memory()
            # 设置内存限制为总内存减去 1GB
            memory_limit_mb = total_memory_mb + 5120  # 1GB = 1024MB
            # 设置虚拟设备配置
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)])
            print(f"memory_limit_mb: {memory_limit_mb} MB")
    except RuntimeError as e:
        print(e)
        
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(2)
tf.random.set_seed(2)

# 读取CSV文件
csv_path = 'unique_compounds.csv'
df = pd.read_csv(csv_path)
with open("zmatrix_data.pkl", "rb") as file:
    zmatrix_data_dict= pickle.load(file)

df['Has_Alcoholic'] = df['Tag'].str.contains('Alcoholic').astype(int)
    
def get_spin_state(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        mol = Chem.AddHs(mol)  # 添加所有隐式氢
        num_radicals = sum(1 for atom in mol.GetAtoms() if atom.GetNumRadicalElectrons() > 0)
        if num_radicals % 2 == 0:
            return 0.0  # 单重态
        else:
            return 1.0  # 双重态
    except Exception as e:
        print(f"Error in spin_state for {smiles}: {e}")
        return None

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output

class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets, zmatrix_data_dict):
        super().__init__(allowable_sets)
        self.zmatrix_data_dict = zmatrix_data_dict  # zmatrix_data_dict 存储了多个 SMILES 对应的 Z-matrix 数据

    def encode(self, atom, smiles):
        # 获取原子特征
        output = super().encode(atom)
        atom_idx = atom.GetIdx()  
        zmatrix_data = self.zmatrix_data_dict.get(smiles, None) 
        if zmatrix_data is not None and atom_idx < len(zmatrix_data):
            coordinates = zmatrix_data[atom_idx] 
            if coordinates is None:
                print(f"Warning: No 'Coordinates' found for atom {atom_idx}. Using default coordinates [0, 0, 0].")
                coordinates = np.zeros(3)  # 如果没有坐标，使用零向量
            coordinates = np.array(coordinates)
            output = np.concatenate((output, coordinates))
        return output

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()

    def is_in_ring(self, atom):
        return 1 if atom.IsInRing() else 0

    def formal_charge(self, atom):
        return atom.GetFormalCharge()

    def degree_of_freedom(self, atom):
        return atom.GetDegree() - 1

    def spin_state(self, atom):
        # 注意这里需要通过整个分子对象来调用 spin_state
        return getattr(atom, 'spin_state', 0.0)  # 默认为0.0表示单重态特征
    
    def covalent_radius(self, atom):
        covalent_radii = {
            'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84,
            'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
            'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07,
            'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
            'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.61,
            'Fe': 1.52, 'Co': 1.50, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
            'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20,
            'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75,
            'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42,
            'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39,
            'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40, 'Cs': 2.44,
            'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
            'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94,
            'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87,
            'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70, 'W': 1.62, 'Re': 1.51,
            'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
            'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40, 'At': 1.50,
            'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06,
            'Pa': 2.00, 'U': 1.96, 'Np': 1.90, 'Pu': 1.87, 'Am': 1.80,
            'Cm': 1.69, 'Bk': 1.54, 'Cf': 1.68, 'Es': 1.65, 'Fm': 1.67,
            'Md': 1.73, 'No': 1.76, 'Lr': 1.61,
        }
        return covalent_radii.get(atom.GetSymbol(), 0.0)

class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1  # 为 none 预留添加位数

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        bond_features = {  # 收集所有特征
            'bond_type': self.bond_type(bond),
            'conjugated': self.conjugated(bond),
            'is_in_ring': self.is_in_ring(bond),
            'bond_type_as_double': self.bond_type_as_double(bond),
            'aromaticity': self.aromaticity(bond),
            'polarity': self.polarity(bond),  # 新增键极性的特征
        }
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = bond_features[name_feature]
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()

    def is_in_ring(self, bond):
        return 1 if bond.IsInRing() else 0

    def bond_type_as_double(self, bond):
        return bond.GetBondTypeAsDouble()

    def aromaticity(self, bond):
        return 1 if bond.GetIsAromatic() else 0
        
    def polarity(self, bond):
        try:
            electronegativity = {
                'H': 2.20, 'He': 0.0, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
                'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 0.0, 'Na': 0.93, 'Mg': 1.31,
                'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 0.0,
                'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
                'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
                'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 3.0,
                'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.6, 'Mo': 2.16,
                'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
                'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 2.6,
                'Cs': 0.79, 'Ba': 0.89, 'La': 1.1, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
                'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.00, 'Gd': 1.20, 'Tb': 1.21, 'Dy': 1.22,
                'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.1, 'Lu': 1.27, 'Hf': 1.3,
                'Ta': 1.5, 'W': 2.36, 'Re': 1.9, 'Os': 2.2, 'Ir': 2.20, 'Pt': 2.28,
                'Au': 2.54, 'Hg': 2.0, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02, 'Po': 2.0,
                'At': 2.2, 'Rn': 0.0, 'Fr': 0.7, 'Ra': 0.9, 'Ac': 1.1, 'Th': 1.3,
                'Pa': 1.5, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28, 'Am': 1.13, 'Cm': 1.28,
                'Bk': 1.3, 'Cf': 1.3, 'Es': 1.3, 'Fm': 1.3, 'Md': 1.3, 'No': 1.3, 'Lr': 1.3,
            }
    
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            symbol1 = atom1.GetSymbol()
            symbol2 = atom2.GetSymbol()
            
            if symbol1 in electronegativity and symbol2 in electronegativity:
                polarity = abs(electronegativity[symbol1] - electronegativity[symbol2])
                return polarity
            else:
                return None
        except Exception as e:
            print(f"Error in getting bond polarity: {e}")
            return None

atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": set("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Sr Zr Mo Ag Cd Sn Sb Te I Ba Hg Pb Bi Rn".split()),
        "n_valence": set(range(0, 7)),
        "n_hydrogens": set(range(0, 5)),
        "hybridization": {"s", "sp", "sp2", "sp3"},
        "is_in_ring": {0, 1},
        "formal_charge": set(range(-3, 4)),
        "degree_of_freedom": set(range(0, 6)),
        "spin_state": {0.0, 1.0},
        "covalent_radius": set([float('-inf'), float('inf')])
    },
    zmatrix_data_dict=zmatrix_data_dict
)
bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},  # 键类型
        "conjugated": {True, False},  # 是否共轭
        "is_in_ring": {0, 1},  # 是否在环中
        "bond_type_as_double": {1.0, 1.5, 2.0, 3.0},  # 键类型数值
        "aromaticity": {0, 1},  # 芳香性
        "polarity": set([float('-inf'), float('inf')])
    }
)

def molecule_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    spin_state_value = get_spin_state(smiles)
    for atom in molecule.GetAtoms():
        atom.spin_state = spin_state_value  # 给每个原子添加 spin_state 属性
    return molecule

def graph_from_molecule(molecule, smiles):  # 接收 smiles 参数
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_feature = atom_featurizer.encode(atom, smiles)  # 使用 smiles
        atom_features.append(atom_feature)

        # 自环
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))  # 自环的键特征

        # 遍历每个邻居
        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    # 返回对象数组
    return np.array(atom_features, dtype=object), np.array(bond_features, dtype=object), np.array(pair_indices, dtype=object)

def graphs_from_smiles(smiles_list):
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for smiles in smiles_list:
        molecule = molecule_from_smiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule, smiles)  # 传递 smiles
        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    )
smiles_list = df['SMILES'].tolist()
atom_features, bond_features, pair_indices = graphs_from_smiles(smiles_list)

x_train = graphs_from_smiles(df[df['dataset'] == 'train']['SMILES'].tolist())
x_valid = graphs_from_smiles(df[df['dataset'] == 'val']['SMILES'].tolist())
x_test = graphs_from_smiles(df[df['dataset'] == 'test']['SMILES'].tolist())

y_train = df[df['dataset'] == 'train']['Has_Alcoholic'].values
y_valid = df[df['dataset'] == 'val']['Has_Alcoholic'].values
y_test = df[df['dataset'] == 'test']['Has_Alcoholic'].values

def prepare_batch(x_batch, y_batch):
    atom_features, bond_features, pair_indices = x_batch

    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    molecule_indices = tf.range(tf.shape(num_atoms)[0])
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)

    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch

def MPNNDataset(X, y, batch_size=1, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)

class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim), initializer="zeros", name="bias",
        )
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 0],
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_features

def create_adjacency_matrix(pair_indices):
    num_atoms = tf.reduce_max(pair_indices) + 1
    adjacency_matrix = tf.zeros((num_atoms, num_atoms), dtype=tf.float32)
    adjacency_matrix = tf.tensor_scatter_nd_update(adjacency_matrix, pair_indices, tf.ones(tf.shape(pair_indices)[0], dtype=tf.float32))
    return adjacency_matrix

class GCNLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer="glorot_uniform",
            name="kernel",
        )

    def call(self, inputs):
        atom_features, adjacency_matrix = inputs
        aggregated_features = tf.matmul(adjacency_matrix, atom_features)
        transformed_features = tf.matmul(aggregated_features, self.kernel)
        return transformed_features

class BondConvolutionLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(BondConvolutionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            name="kernel",
        )

    def call(self, bond_features):
        # 对键特征进行卷积操作
        transformed_bond_features = tf.matmul(bond_features, self.kernel)
        return transformed_bond_features


class MessagePassing(layers.Layer):
    def __init__(self, units, steps=16, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        gcn_features, transformed_bond_features, pair_indices = inputs
        atom_features_updated = tf.pad(gcn_features, [(0, 0), (0, self.pad_length)])
        for i in range(self.steps):
            atom_features_aggregated = self.message_step(
                [atom_features_updated, transformed_bond_features, pair_indices]
            )
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated

class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        atom_features, molecule_indicator = inputs
        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)

class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self, num_heads=16, embed_dim=128, dense_dim=512, batch_size=1, output_dim=64, **kwargs
    ):
        super().__init__(**kwargs)
        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(output_dim),  # 修改为 output_dim
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        attention_output, attention_scores = self.attention(x, x, attention_mask=padding_mask, return_attention_scores=True)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        pooled_output = self.average_pooling(proj_output)
        
        # Normalize attention scores
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        print(f"attention_scores shape after attention: {attention_scores.shape}")

        return pooled_output, attention_scores


class LabelWiseLossHistory(callbacks.Callback):
    def __init__(self, valid_dataset):
        super(LabelWiseLossHistory, self).__init__()
        self.valid_dataset = valid_dataset
        self.labelwise_losses = []

    def on_epoch_end(self, epoch, logs=None):
        x_valid = []
        y_valid = []
        
        # 确保所有数据都是 float32 类型
        for x_batch, y_batch in self.valid_dataset:
            x_valid.append(tf.cast(x_batch, tf.float32))
            y_valid.append(tf.cast(y_batch, tf.float32))
        
        x_valid = tf.concat(x_valid, axis=0)
        y_valid = tf.concat(y_valid, axis=0)

        # 预测并计算损失
        y_pred = self.model.predict(x_valid)
        labelwise_loss = tf.keras.losses.binary_crossentropy(y_valid, y_pred)
        
        # 记录每个标签的平均损失
        mean_labelwise_loss = np.mean(labelwise_loss.numpy(), axis=0)
        self.labelwise_losses.append(mean_labelwise_loss)
        
        # 将每个标签的损失添加到 logs 中
        for i, loss in enumerate(mean_labelwise_loss):
            logs[f'label_{i}_loss'] = loss

def MPNNModel(
    atom_dim,
    bond_dim,
    output_dim,
    batch_size=1,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
    dropout_rate=0.5
):

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")

    adjacency_matrix = create_adjacency_matrix(pair_indices)
    print("atom_features shape:", atom_features.shape)
    print("adjacency_matrix shape:", adjacency_matrix.shape)
    gcn_features = GCNLayer(message_units)([atom_features, adjacency_matrix])

    bond_convolution = BondConvolutionLayer(message_units)
    transformed_bond_features = bond_convolution(bond_features)

    x = MessagePassing(message_units, message_steps)(
        [gcn_features, transformed_bond_features, pair_indices]
    )
    
    x, attention_scores = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])
    
    x = layers.Dropout(dropout_rate)(x)
    dense_output = layers.Dense(dense_units, activation="relu")(x)
    final_output = layers.Dense(output_dim, activation="sigmoid")(dense_output)
    
    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, molecule_indicator],
        outputs=[final_output]
    )
    return model
def compute_gradients(model, inputs, target_class_index, num_atoms):
    atom_features, bond_features, pair_indices, molecule_indicator = inputs
    inputs_dict = {
        'atom_features': atom_features,
        'bond_features': bond_features,
        'pair_indices': pair_indices,
        'molecule_indicator': molecule_indicator,
    }

    with tf.GradientTape() as tape:
        tape.watch(inputs_dict['atom_features'])
        predictions = model(inputs_dict) 
        print("Predictions shape:", predictions.shape)
        loss = predictions[:, target_class_index]
        print("Loss shape:", loss.shape)

    # 计算梯度
    gradients = tape.gradient(loss, inputs_dict['atom_features'])
    print("Gradients shape:", gradients.shape)

    # 确保 num_atoms 是一个一维的张量
    num_atoms = tf.constant(num_atoms, dtype=tf.int64)

    # 将梯度转换为 RaggedTensor
    gradients = tf.RaggedTensor.from_row_lengths(gradients, num_atoms)

    # 打印梯度的形状
    print("Ragged Gradients shape:", gradients.shape)

    return gradients

def normalize_gradients(gradients):
    if isinstance(gradients, tf.RaggedTensor):
        gradients = gradients.flat_values

    min_val = tf.reduce_min(gradients)
    max_val = tf.reduce_max(gradients)
    normalized_gradients = (gradients - min_val) / (max_val - min_val)
    node_importance = tf.reduce_mean(normalized_gradients, axis=1)
    print("Node Importance shape:", node_importance.shape)
    return node_importance

def visualize_importance(inputs, gradients, target_class_index, output_file='importance_data.csv'):
    atom_features, bond_features, pair_indices, molecule_indicator = inputs

    # 将梯度归一化
    normalized_gradients = normalize_gradients(gradients)

    # 将节点特征和对原子对的索引转换为 NumPy 数组
    atom_features = atom_features.numpy()
    pair_indices = pair_indices.numpy()
    normalized_gradients = normalized_gradients.numpy()

    # 创建一个 NetworkX 图
    G = nx.Graph()

    # 添加节点及其梯度
    for i in range(len(atom_features)):
        node_gradient = normalized_gradients[i]
        G.add_node(i, importance=node_gradient)

    # 添加边
    for pair in pair_indices:
        G.add_edge(pair[0], pair[1])

    # 绘制图
    pos = nx.spring_layout(G)
    node_importance = nx.get_node_attributes(G, 'importance')
    nx.draw(G, pos, node_color=[node_importance[node] for node in G.nodes],
            cmap=plt.cm.viridis, node_size=2000, with_labels=True)
    plt.title(f"Node Importance for Target Class {target_class_index}")
    plt.show()

    node_data = [{'node_id': node, 'importance': node_importance[node]} for node in G.nodes]
    edge_data = [{'source': edge[0], 'target': edge[1]} for edge in G.edges]

    # 将数据作为返回值输出
    return node_data, edge_data
output_dim = 1
positive_samples_ratio = 1-np.mean(y_train)
alpha = positive_samples_ratio
print(f'alpha={alpha}')
alpha = 0.5

mpnn = MPNNModel(
    atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0], output_dim=output_dim
)

def focal_loss(alpha=0.25, gamma=0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = K.cast(y_true, 'float32')
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        loss = alpha_t * K.pow(1 - p_t, gamma) * cross_entropy
        loss = K.mean(loss, axis=-1)
        return loss
    return focal_loss_fixed

mpnn.compile(
    loss=focal_loss(alpha=alpha, gamma=0),
    optimizer=keras.optimizers.Adam(learning_rate=0.000006),
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
)
mpnn.summary()

keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)
train = 0
testa = np.abs(1-train)
train_dataset = MPNNDataset(x_train, y_train)
valid_dataset = MPNNDataset(x_valid, y_valid)
test_dataset = MPNNDataset(x_test, y_test)
checkpoint_dir = "./241219-Alcoholic-youhua"

if train == 1:
    # 初始化全局变量
    best_validation_loss = np.inf
    checkpoint_dir = "./241214-Alcoholic-youhua2"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 定义目标函数
    def objective(trial):
        global best_validation_loss
        
        # 从 Optuna 的试验中获取超参数
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        num_attention_heads = trial.suggest_categorical("num_attention_heads", [8, 16, 32, 64])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.7)
        dense_units = trial.suggest_categorical("dense_units", [64, 128, 256, 512])
        message_steps = trial.suggest_categorical("message_steps", [4, 8, 16, 32, 64])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        
        # 创建一个新的文件夹来保存本次尝试的结果
        trial_dir = os.path.join(checkpoint_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # 使用传入的超参数初始化模型
        mpnn = MPNNModel(
            atom_dim=x_train[0][0][0].shape[0], 
            bond_dim=x_train[1][0][0].shape[0], 
            output_dim=output_dim,
            message_steps=message_steps, 
            num_attention_heads=num_attention_heads, 
            dense_units=dense_units,
            dropout_rate=dropout_rate
        )
        
        # 使用传入的学习率和 alpha 来编译模型
        mpnn.compile(
            loss=focal_loss(alpha=alpha, gamma=0),
            optimizer=Adam(learning_rate=learning_rate),
            metrics=[BinaryAccuracy(name="accuracy")]
        )
        
        # 准备数据集
        train_dataset = MPNNDataset(x_train, y_train, batch_size=batch_size)
        valid_dataset = MPNNDataset(x_valid, y_valid, batch_size=batch_size)
        
        # 定义检查点路径
        checkpoint_filepath = os.path.join(trial_dir, "model_epoch_{epoch:02d}_val_loss_{val_loss:.8f}.h5")
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=30,
            verbose=2,
            restore_best_weights=True
        )
        
        # 训练模型
        history = mpnn.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=500,
            verbose=2,
            callbacks=[model_checkpoint_callback, early_stopping_callback]
        )
        
        # 保存训练历史
        history_filepath = os.path.join(trial_dir, "trainstory.csv")
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(history_filepath, index=False)
        
        # 保存本次尝试的参数
        params_filepath = os.path.join(trial_dir, "params.txt")
        with open(params_filepath, "w") as f:
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"Number of Attention Heads: {num_attention_heads}\n")
            f.write(f"Dropout Rate: {dropout_rate}\n")
            f.write(f"Dense Units: {dense_units}\n")
            f.write(f"Message Steps: {message_steps}\n")
            f.write(f"Batch Size: {batch_size}\n")
        
        # 获取验证损失
        val_loss = np.min(history.history['val_loss'])
        print(f"Validation loss: {val_loss}")
        
        # 动态早停
        trial.report(val_loss, step=1)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return val_loss
    
    # 设置 Optuna 的优化器
    pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=2, interval_steps=1)
    study = optuna.create_study(direction="minimize", study_name="MPNN Hyperparameter Optimization", pruner=pruner)
    
    # 添加 TensorBoard 回调
    tensorboard_callback = TensorBoardCallback("./optuna_logs", metric_name="val_loss")
    
    try:
        study.optimize(objective, n_trials=100, callbacks=[tensorboard_callback])
    except KeyboardInterrupt:
        print("Optimization interrupted manually.")
    
    # 保存最佳参数
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Parameters: {study.best_trial.params}")
    print(f"Best Validation Loss: {study.best_value}")
    
    # 保存结果到文件
    with open(os.path.join(checkpoint_dir, "best_trial.txt"), "w") as f:
        f.write(f"Best Trial: {study.best_trial.number}\n")
        f.write(f"Best Parameters: {study.best_trial.params}\n")
        f.write(f"Best Validation Loss: {study.best_value}\n")
    
    # 可视化
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_optimization_history(study).show()
        
if testa == 1:
    output_dim = 1  # 修改为单分类任务
    best_params = {
        'learning_rate': 6.571220684252959e-05,
        'num_attention_heads': 32,
        'dropout_rate':  0.5993701668192596,
        'dense_units': 256,
        'message_steps': 4,
        'batch_size': 1
    }
    
    mpnn = MPNNModel(
        atom_dim=x_train[0][0][0].shape[0],
        bond_dim=x_train[1][0][0].shape[0],
        output_dim=output_dim,
        message_steps=best_params['message_steps'], 
        num_attention_heads=best_params['num_attention_heads'], 
        dense_units=best_params['dense_units'],
        dropout_rate=best_params['dropout_rate']
    )
    
    # 加载最佳权重文件
    checkpoint_filepath = os.path.join(checkpoint_dir, "model_epoch_68_val_loss_0.06237833.h5")  # 替换为你的最佳权重文件
    mpnn.load_weights(checkpoint_filepath)
    mpnn.compile(
        loss='binary_crossentropy',  # 使用二分类交叉熵损失函数
        optimizer=keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    # 生成验证集的预测结果
    y_pred = mpnn.predict(valid_dataset)
    y_pred_binary = (y_pred > 0.5).astype(int)  # 单分类任务

    # 标记错误样本
    valid_errors = (y_valid != y_pred_binary).astype(int)

    # 设置输出文件路径
    output_file_path = os.path.join(checkpoint_dir, 'output.txt')

    with open(output_file_path, 'w') as f, redirect_stdout(f):
        print(f'y_pred_binary shape: {y_pred_binary.shape}')
        print(f'Sample y_pred_binary: {y_pred_binary[:5]}')
        print(f'y_valid shape: {y_valid.shape}')
        print(f'Sample y_valid: {y_valid[:5]}')

        label_name = "Alcoholic"

        # 保存验证集的综合分类报告为CSV文件
        comprehensive_report_filepath = os.path.join(checkpoint_dir, "comprehensive_classification_report_valid.csv")
        report = classification_report(
            y_valid, y_pred_binary,
            target_names=["Not " + label_name, label_name],
            output_dict=True,
            labels=[0, 1]  # 确保包含所有标签
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(comprehensive_report_filepath, index=True)
        print(f"Comprehensive classification report (validation) saved to {comprehensive_report_filepath}")

        # 生成验证集的预测结果CSV文件
        valid_results_df = df[df['dataset'] == 'val'].copy()
        valid_results_df['Alcoholic-ypred'] = y_pred  # 保存预测值（非1/0）
        valid_results_df['Predicted_Binary'] = y_pred_binary  # 保存预测的二分类结果
        valid_results_df['True_Label'] = y_valid  # 保存真实标签
        valid_results_df.reset_index(inplace=True)  # 重置索引
        valid_results_csv_filepath = os.path.join(checkpoint_dir, "Alcoholic-valid-result.csv")
        valid_results_df.to_csv(valid_results_csv_filepath, index=False)
        print(f"Validation results saved to {valid_results_csv_filepath}")

        # 生成验证集错误样本的CSV文件
        valid_errors_df = valid_results_df[valid_results_df['True_Label'] != valid_results_df['Predicted_Binary']]  # 仅保留错误样本
        valid_errors_csv_filepath = os.path.join(checkpoint_dir, "validation_errors.csv")
        valid_errors_df.to_csv(valid_errors_csv_filepath, index=False)
        print(f"Validation errors saved to {valid_errors_csv_filepath}")

        # 生成测试集的预测结果
        y_pred_test = mpnn.predict(test_dataset)
        y_pred_binary_test = (y_pred_test > 0.5).astype(int)

        # 标记错误样本
        test_errors = (y_test != y_pred_binary_test).astype(int)

        print(f'Test y_pred_binary shape: {y_pred_binary_test.shape}')
        print(f'Sample Test y_pred_binary: {y_pred_binary_test[:5]}')
        print(f'y_test shape: {y_test.shape}')
        print(f'Sample y_test: {y_test[:5]}')

        # 保存测试集的综合分类报告为CSV文件
        comprehensive_report_filepath_test = os.path.join(checkpoint_dir, "comprehensive_classification_report_test.csv")
        report_test = classification_report(
            y_test, y_pred_binary_test,
            target_names=["Not " + label_name, label_name],
            output_dict=True,
            labels=[0, 1]  # 确保包含所有标签
        )
        report_df_test = pd.DataFrame(report_test).transpose()
        report_df_test.to_csv(comprehensive_report_filepath_test, index=True)
        print(f"Comprehensive classification report (test) saved to {comprehensive_report_filepath_test}")

        # 生成测试集的预测结果CSV文件
        test_results_df = df[df['dataset'] == 'test'].copy()
        test_results_df['Alcoholic-ypred'] = y_pred_test  # 保存预测值（非1/0）
        test_results_df['Predicted_Binary'] = y_pred_binary_test  # 保存预测的二分类结果
        test_results_df['True_Label'] = y_test  # 保存真实标签
        test_results_df.reset_index(inplace=True)  # 重置索引
        test_results_csv_filepath = os.path.join(checkpoint_dir, "Alcoholic-test-result.csv")
        test_results_df.to_csv(test_results_csv_filepath, index=False)
        print(f"Test results saved to {test_results_csv_filepath}")

        # 生成测试集错误样本的CSV文件
        test_errors_df = test_results_df[test_results_df['True_Label'] != test_results_df['Predicted_Binary']]  # 仅保留错误样本
        test_errors_csv_filepath = os.path.join(checkpoint_dir, "test_errors.csv")
        test_errors_df.to_csv(test_errors_csv_filepath, index=False)
        print(f"Test errors saved to {test_errors_csv_filepath}")

    def create_plot_dataset(df, num_samples=1):
        sample_indices = np.random.choice(df.index, num_samples, replace=False)
        smiles_list = df.loc[sample_indices, 'SMILES'].tolist()
        y_list = df.loc[sample_indices, 'Has_Alcoholic'].values
        x_plot = graphs_from_smiles(smiles_list)
        
        # 打印抽取的样本的 SMILES
        for index, smiles in zip(sample_indices, smiles_list):
            print(f"Sample Index: {index}, SMILES: {smiles}")
        
        return x_plot, y_list, sample_indices
    x_plot, y_plot, plot_indices = create_plot_dataset(df, num_samples=1)
    
    def create_plot_dataset(df, indices):
        smiles_list = df.loc[indices, 'SMILES'].tolist()
        y_list = df.loc[indices, 'Has_Alcoholic'].values
        x_plot = graphs_from_smiles(smiles_list)
        num_atoms_list = []
        # 计算每个样本的原子数量
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            num_atoms = mol.GetNumHeavyAtoms()
            num_atoms_list.append(num_atoms)

        
        # 打印抽取的样本的 SMILES
        for index, smiles in zip(indices, smiles_list):
            print(f"Sample Index: {index}, SMILES: {smiles}")
        
        return x_plot, y_list, indices, num_atoms_list
    
    specified_indices = [2442, 6071, 626]  # 指定你想要的多个样本索引
    
    # 遍历每个索引
    for index in specified_indices:
        x_plot, y_plot, plot_indices, num_atoms_list = create_plot_dataset(df, [index])
        plot_dataset = MPNNDataset(x_plot, y_plot, batch_size=1, shuffle=False)
    
        for inputs, y_plot in plot_dataset.take(1):
            break
    
        # 检查 inputs 是否为元组
        print("Inputs type:", type(inputs))
        print("Inputs shape:", [i.shape for i in inputs])
    
        # 选择一个目标类别
        target_class_index = 0
    
        # 计算梯度
        gradients = compute_gradients(mpnn, inputs, target_class_index, num_atoms=num_atoms_list)
    
        # 可视化重要性分布
        node_data, edge_data = visualize_importance(inputs, gradients, target_class_index)
    
        # 生成文件名
        node_importance_filename = f'node_importance_{index}.csv'
        edge_data_filename = f'edge_data_{index}.csv'
    
        # 保存节点数据到 CSV 文件
        with open(os.path.join(checkpoint_dir, node_importance_filename), 'w', newline='') as csvfile:
            fieldnames = ['node_id', 'importance']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(node_data)
    
        # 保存边数据到 CSV 文件
        with open(os.path.join(checkpoint_dir, edge_data_filename), 'w', newline='') as csvfile:
            fieldnames = ['source', 'target']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(edge_data)
    
        print(f"数据已保存到指定目录：{checkpoint_dir}")
        print(f"节点重要性数据文件名：{node_importance_filename}")
        print(f"边数据文件名：{edge_data_filename}")