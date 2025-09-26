import pandas as pd
import numpy as np
import os
import sys

# 保持不变
def get_input_variables():
    """定义输入变量的名称。"""
    return ['M', 'R1', 'R2', 'R3', 'T', 'P', 'Al/M', 'Time', 'Cat', 'Cocat']

def get_output_variables():
    """定义输出变量的名称。"""
    return ['CA', 'MW', 'MWD']

# 保持不变
def get_categorical_mappings():
    """
    此映射字典不再用于数据加载，但仍用于 app.py 中的**用户输入界面**和**反向查找**。
    键名为小写。
    """
    return {
        'M': {'ti': 22, 'zr': 40, 'hf': 72},
        'R1': {
            'phenyl': 20.5, '2,3,4,5,6-pentafluorophenyl': 21.1, 'cyclooctyl': 20.1,
            'cycloheptyl': 20.1, '2-methylphenyl': 25.5, '2-isopropylphenyl': 27.6,
            'cyclohexyl': 19.0, 'cumyl': 7.1, '2-phenylethyl': 20.5, '3-phenylpropyl': 20.3,
            '2,4,6-trifluorophenyl': 21.1, '2,6-difluorophenyl': 21.1, '2-fluorophenyl': 19.5,
            '3,4,5-trifluorophenyl': 18.8, '3,5-difluorophenyl': 18.8, '4-fluorophenyl': 18.8,
            '4-(4-vinylphenyl)-2,6-difluorophenyl': 21.2, '4-(4-ethylphenyl)-2,6-difluorophenyl': 21.2,
            '2,6-dimethylphenyl': 29.3, '2,6-diisopropylphenyl': 37.4, 'naphthyl': 19.9,
            'methyl': 16.5, 'cyclopropyl': 16.8, 'cyclopentyl': 23.1, 'ethyl': 16.9, 'propyl': 19.4,
            'isopropyl': 18.8, 'n-hexyl': 17.7, 'cyclobutyl': 17.5, '2-methylcyclohexyl': 19.4,
            'adamantyl': 2.9, '2-tert-butylphenyl': 29.8, '3,5-di-tert-butylphenyl': 20.7,
            '4-tert-butylphenyl': 19.1,
        },
        'R2': {
            'tert-butyl': 8.4, 'methyl': 1.9, 'h': 0.1, 'phenyl': 2.1, 'trimethylsilyl': 6.3,
            'cyclopentyl': 2.5, 'isopropyl': 4.7, 'cumyl': 7.1, 'cl': 1.1, 'br': 1.7,
            'adamantyl': 2.9, 'f': 0.9, 'i': 2.8,
        },
        'R3': {
            'h': 6.0, 'tert-butyl': 5.0, 'methyl': 4.0, 'i': 6.0, 'cl': 7.0, 'br': 8.0, 'phenyl': 2.0, 'cumyl': 3.0, 'methoxy': 1.0
        }
    }

def load_data():
    """
    加载并处理所有数据文件。**修正：** M, R1, R2, R3 已被视为数值输入，不再进行映射。
    """
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.getcwd() 
    
    data_dict = {}
    features = get_input_variables()
    
    # 🌟 关键修正：不再强制读取为字符串
    
    for target in get_output_variables():
        train_data_path = os.path.join(base_dir, 'Models', target, f'train_data_{target}.csv')
        test_data_path = os.path.join(base_dir, 'Models', target, f'test_data_{target}.csv')

        if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
            raise FileNotFoundError(f"找不到 {target} 的数据文件。请确保 Models/{target} 目录下有 train_data_{target}.csv 和 test_data_{target}.csv 文件。")

        # 使用默认的 dtype 推断，确保 M, R1, R2, R3 被识别为数值
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        
        # 确保所有特征列都被正确地转换为数值类型，防止任何残留的字符串
        for col in features:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
            
        train_df[target] = pd.to_numeric(train_df[target], errors='coerce')
        test_df[target] = pd.to_numeric(test_df[target], errors='coerce')
        

        data_dict[target] = {
            'train_x': train_df[features],
            'train_y': train_df[target],
            'test_x': test_df[features],
            'test_y': test_df[target]
        }
    
    return data_dict

# get_smiles_mappings() 保持不变 (用于 app.py 中的结构绘图)
def get_smiles_mappings():
    return {
        'R1': {
            'phenyl': 'c1ccccc1', '2,3,4,5,6-pentafluorophenyl': 'C1=C(C(=C(C(=C1F)F)F)F)F', 'cyclooctyl': 'C1CCCCCCC1',
            'cycloheptyl': 'C1CCCCCC1', '2-methylphenyl': 'Cc1ccccc1', '2-isopropylphenyl': 'CC(C)c1ccccc1',
            'cyclohexyl': 'C1CCCCC1', 'cumyl': 'CC(C)(c1ccccc1)c1ccccc1', '2-phenylethyl': 'c1ccccc1CC', '3-phenylpropyl': 'c1ccccc1CCC',
            '2,4,6-trifluorophenyl': 'Fc1cc(F)cc(F)c1', '2,6-difluorophenyl': 'Fc1cccc(F)c1', '2-fluorophenyl': 'Fc1ccccc1',
            '3,4,5-trifluorophenyl': 'Fc1cc(F)c(F)cc1', '3,5-difluorophenyl': 'Fc1cc(F)ccc1', '4-fluorophenyl': 'Fc1ccc(F)cc1',
            '4-(4-vinylphenyl)-2,6-difluorophenyl': 'C=Cc1ccc(cc1)c1c(F)cc(F)c(c1)F', '4-(4-ethylphenyl)-2,6-difluorophenyl': 'CCc1ccc(cc1)c1c(F)cc(F)c(c1)F',
            '2,6-dimethylphenyl': 'Cc1cccc(C)c1', '2,6-diisopropylphenyl': 'CC(C)c1cccc(C(C)C)c1', 'naphthyl': 'c1ccc2c(c1)cccc2',
            'methyl': 'C', 'cyclopropyl': 'C1CC1', 'cyclopentyl': 'C1CCCC1', 'ethyl': 'CC', 'propyl': 'CCC',
            'isopropyl': 'CC(C)', 'n-hexyl': 'CCCCCC', 'cyclobutyl': 'C1CCC1', '2-methylcyclohexyl': 'CC1CCCCC1',
            'adamantyl': 'C1C2CC3CC1C(C2)C3', '2-tert-butylphenyl': 'CC(C)(C)c1ccccc1', '3,5-di-tert-butylphenyl': 'CC(C)(C)c1cc(cc(c1)C(C)(C))',
            '4-tert-butylphenyl': 'CC(C)(C)c1ccc(C(C)(C))cc1',
        },
        'R2': {
            'tert-butyl': 'C(C)(C)C', 'methyl': 'C', 'h': '[H]', 'phenyl': 'c1ccccc1', 'trimethylsilyl': '[Si](C)(C)C',
            'cyclopentyl': 'C1CCCC1', 'isopropyl': 'C(C)C', 'cumyl': 'C(C)(C)c1ccccc1', 'cl': '[Cl]', 'br': '[Br]',
            'adamantyl': 'C1C2CC3CC1C(C2)C3', 'f': '[F]', 'i': '[I]',
        },
        'R3': {
            'h': '[H]', 'tert-butyl': 'C(C)(C)C', 'methyl': 'C', 'i': '[I]', 'cl': '[Cl]', 'br': '[Br]', 'phenyl': 'c1ccccc1', 'cumyl': 'CC(C)(c1ccccc1)c1ccccc1', 'methoxy': 'CO'
        }
    }