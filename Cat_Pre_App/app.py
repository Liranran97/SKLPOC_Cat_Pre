import sys # <-- 新增的导入
import streamlit as st
import streamlit.web.cli as stcli
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, get_input_variables, get_output_variables, get_categorical_mappings, get_smiles_mappings
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import base64
from io import BytesIO
import itertools

# --- 🎯 路径修正：将当前工作目录切换到 app.py 所在的文件夹 ---
# 这确保了 'Models/Trained' 这样的相对路径是正确的
# --- 🎯 路径修正：将当前工作目录切换到 app.py 所在的文件夹 ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    import logging
    logging.warning(f"当前工作目录已切换至: {os.getcwd()}")
    st.write(f"当前工作目录已切换至: {os.getcwd()}") 
except Exception as e:
    print(f"无法切换工作目录{e}") 

# 加载模型、标准化器和数据
@st.cache_resource
def load_models_and_data():
    """
    加载所有已训练的模型、标准化器和原始数据。
    同时构建用于数据可视化的 combined_df，并执行反向映射。
    """
    models = {}
    scalers_X = {}
    scalers_y = {}
    metrics = {}
    
    model_dir = 'Models/Trained'

    if not os.path.exists(model_dir):
        st.error("找不到训练好的模型。请先运行 main.py。")
        st.stop()
        
    data_dict = load_data() # data_dict 中的 R1, R2, R3, M 已经是数值

    for target in get_output_variables():
        model_path = os.path.join(model_dir, f'model_{target}.joblib')
        scaler_X_path = os.path.join(model_dir, f'scaler_X_{target}.joblib')
        scaler_y_path = os.path.join(model_dir, f'scaler_y_{target}.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
            models[target] = joblib.load(model_path)
            scalers_X[target] = joblib.load(scaler_X_path)
            scalers_y[target] = joblib.load(scaler_y_path)

            # ------------------------------------------------------------------
            # 计算并存储性能指标和残差标准差 (使用数值化数据)
            # ------------------------------------------------------------------
            # 注意：这里使用 load_data() 返回的 train_x/train_y，它们是数值化的
            X_train_scaled = scalers_X[target].transform(data_dict[target]['train_x'])
            y_pred_scaled = models[target].predict(X_train_scaled)
            
            y_pred = scalers_y[target].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = data_dict[target]['train_y'].values.flatten()
            
            residuals = y_true - y_pred
            
            metrics[target] = {
                'R2': r2_score(y_true, y_pred),
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'Residual_Std': np.std(residuals)
            }
        else:
            st.error(f"缺少 {target} 的模型或标准化器文件。")
            st.stop()
            
    # ------------------------------------------------------------------
    # 构建用于可视化的 combined_df (需要将分类变量从数值反向映射回名称)
    # ------------------------------------------------------------------
    all_dfs = []
    output_vars = get_output_variables()
    
    for target in output_vars:
        # 复制数值化的训练集
        train_df = data_dict[target]['train_x'].copy() 
        train_df[target] = data_dict[target]['train_y']
        
        # 关键：为合并做准备，确保其他目标列存在且为 NaN
        for other_target in output_vars:
            if other_target not in train_df.columns:
                train_df[other_target] = np.nan
                
        all_dfs.append(train_df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=get_input_variables()).reset_index(drop=True)
    
    # 🌟 关键修正：将 M, R1, R2, R3 的数值反向映射回字符串
    # ... (此处应是你之前修正的 M, R1, R2, R3 的反向映射代码)
    cat_mappings = get_categorical_mappings()
    categorical_vars = cat_mappings.keys()
    for var in categorical_vars:
        if var in combined_df.columns:
            inverse_map = {v: k.lower() for k, v in cat_mappings[var].items()}
            combined_df[var] = combined_df[var].map(inverse_map)
            combined_df[var] = combined_df[var].fillna('None')

    # 🌟 新增修正：处理输出变量 (CA, MW, MWD) 中的 NaN
    # 将输出变量中的 NaN 填充为 '-' 字符串，以便 Streamlit 正确显示
    output_vars = get_output_variables() 
    
    for col in output_vars:
        if col in combined_df.columns:
            # 1. 确保它是数值 (float) 类型，这会将所有空值统一为 np.nan
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce') 
            
            # 2. **关键：** 使用 fillna() 将 np.nan 统一替换为字符串 '-'
            #    这一步将所有数值 NaN 转换为字符串。
            combined_df[col] = combined_df[col].fillna('-')
            
            # 3. 将整个列转换为字符串 (如果还没有转换的话)
            combined_df[col] = combined_df[col].astype(str)
            
            # 4. **终极清理：** 将所有 Python None 对象彻底替换为字符串 '-'
            #    这针对的是在数据合并过程中，可能遗留在 object 类型中的 None 值。
            combined_df[col] = combined_df[col].apply(lambda x: '-' if x is None else x)

    # 此时，combined_df 的所有 object 列都只包含字符串，不再包含 None 对象。
        
    return models, scalers_X, scalers_y, combined_df, metrics

def mol_to_base64(mol):
    """将 RDKit 分子对象转换为 Base64 字符串，以便在 Streamlit 中显示。"""
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(300, 200)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    except Exception as e:
        st.warning(f"无法使用 GetDrawingText()，正在尝试 GetSVG()。错误: {e}")
        try:
            svg = drawer.GetSVG()
        except Exception as e2:
            st.error(f"无法生成SVG图像。错误: {e2}")
            return ""

    b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f'data:image/svg+xml;base64,{b64}'

def display_ligand_images(selected_ligands):
    """单独的界面用于显示所选配体的图片"""
    st.header('已选择配体的分子结构')
    st.write('请检查以下分子结构，以确保它们符合您的期望。')
    
    smiles_mappings = get_smiles_mappings()
    
    cols = st.columns(3)
    
    for i, var in enumerate(['R1', 'R2', 'R3']):
        with cols[i]:
            st.subheader(f'{var} 配体')
            ligand_name = selected_ligands.get(var)
            
            if ligand_name and isinstance(ligand_name, str) and ligand_name in smiles_mappings.get(var, {}):
                try:
                    smiles = smiles_mappings[var][ligand_name]
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol:
                        Chem.SanitizeMol(mol)
                        mol = Chem.RemoveHs(mol)
                        img_b64 = mol_to_base64(mol)
                        st.image(img_b64, caption=ligand_name, width='stretch')
                    else:
                        st.warning(f"无法为配体 '{ligand_name}' 生成分子图，其 SMILES 字符串可能无效。")
                        st.write(f"无效的 SMILES: `{smiles}`")
                except KeyError:
                    st.error(f"在 smiles_mappings 中找不到配体 '{ligand_name}'。")
            else:
                st.info("未找到匹配的配体结构。")

#----------------------------------------------------------------------------------------------------------------------------------------------------------

def get_user_input():
    st.sidebar.header('输入参数')
    
    cat_mappings = get_categorical_mappings()
    
    user_inputs = {}
    selected_ligand_names = {}
    
    for var in ['R1', 'R2', 'R3']:
        st.sidebar.subheader(f'选择 {var} 配体')
        options = list(cat_mappings[var].keys())
        selected_option_name = st.sidebar.selectbox(f"请选择 {var} 配体", options)
        selected_ligand_names[var] = selected_option_name
        user_inputs[var] = cat_mappings[var][selected_option_name]

    if 'M' in get_input_variables():
        st.sidebar.subheader('选择 M 金属')
        options = list(cat_mappings['M'].keys())
        selected_option = st.sidebar.selectbox('M', options)
        user_inputs['M'] = cat_mappings['M'][selected_option]
    
    st.sidebar.subheader('输入其他数值参数')
    if 'T' in get_input_variables():
        user_inputs['T'] = st.sidebar.slider('T (°C)', 25.0, 250.0, 80.0)
    if 'P' in get_input_variables():
        user_inputs['P'] = st.sidebar.slider('P (MPa)', 0.1, 1.0, 0.4)
    if 'Al/M' in get_input_variables():
        user_inputs['Al/M'] = st.sidebar.slider('Al/M', 25.0, 12500.0, 2000.0)
    if 'Time' in get_input_variables():
        user_inputs['Time'] = st.sidebar.slider('Time (min)', 1.0, 60.0, 30.0)
    if 'Cat' in get_input_variables():
        user_inputs['Cat'] = st.sidebar.slider('Cat (μmol)', 0.1, 5.0, 1.0)
    if 'Cocat' in get_input_variables():
        user_inputs['Cocat'] = st.sidebar.slider('Cocat (μmol)', 0.1, 10.0, 2.0)
            
    ordered_inputs = {var: [user_inputs[var]] for var in get_input_variables()}
    input_df = pd.DataFrame(ordered_inputs, columns=get_input_variables())
    
    return input_df, selected_ligand_names


def main_forward_prediction(models, scalers_X, scalers_y, metrics):
    """正向预测页面"""
    st.title('FI 催化剂性能预测应用')
    st.write('在左侧输入参数，预测乙烯均聚反应的 CA、MW 和 MWD。')
    
    user_data, selected_ligand_names = get_user_input()
    
    st.subheader('您的输入参数：')
    st.write(user_data)
    
    display_ligand_images(selected_ligand_names)
    
    if st.sidebar.button('进行预测'):
        st.subheader('预测结果：')
        results = {}
        uncertainties = {}
        for target in get_output_variables():
            scaler_X = scalers_X[target]
            scaler_y = scalers_y[target]
            
            scaled_input = scaler_X.transform(user_data)
            prediction_scaled = models[target].predict(scaled_input)
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
            results[target] = prediction[0][0]
            
            # 计算预测不确定性
            uncertainty = metrics[target]['Residual_Std'] * 2  # 简单使用2倍标准差作为95%置信区间
            uncertainties[target] = uncertainty
            
        unit_map = {
            'CA': 'g·mmol⁻¹·h⁻¹',
            'MW': 'kg·mol⁻¹',
            'MWD': '(无单位)'
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label=f"CA ({unit_map['CA']})", value=f"{results['CA']:.2f}", delta=f"±{uncertainties['CA']:.2f}")
        with col2:
            st.metric(label=f"MW ({unit_map['MW']})", value=f"{results['MW']:.2f}", delta=f"±{uncertainties['MW']:.2f}")
        with col3:
            st.metric(label=f"MWD ({unit_map['MWD']})", value=f"{results['MWD']:.2f}", delta=f"±{uncertainties['MWD']:.2f}")


def main_inverse_search(models, scalers_X, scalers_y):
    """
    逆向寻找相似条件页面，通过模型预测生成虚拟数据集进行搜索。
    
    核心逻辑：用户输入目标性能 (CA, MW, MWD)，算法在虚拟数据集的预测
    结果中，寻找性能最接近目标的 5 个输入条件。
    """
    st.title('寻找相似实验条件')
    st.write('输入您期望的 CA、MW 和 MWD，我们将为您寻找最接近的潜在实验条件。')

    st.sidebar.header('输入目标值')

    # 目标性能滑块 (保持不变)
    target_ca = st.sidebar.slider('目标 CA', min_value=0.01, max_value=600.0, value=100.0, key='target_ca_slider')
    target_mw = st.sidebar.slider('目标 MW', min_value=0.1, max_value=130.0, value=30.0, key='target_mw_slider')
    target_mwd = st.sidebar.slider('目标 MWD', min_value=0.1, max_value=6.5, value=3.0, key='target_mwd_slider')

    if st.sidebar.button('开始寻找'):
        st.subheader('搜索结果：')
        
        # 步骤 1: 生成虚拟数据集
        st.info("正在利用模型生成虚拟实验条件数据集，这可能需要一些时间...")
        
        cat_mappings = get_categorical_mappings()
        # smiles_mappings = get_smiles_mappings() # 未使用，可删除

        # 组合所有可能的离散变量
        discrete_vars_combos = list(itertools.product(
            list(cat_mappings['M'].keys()),
            list(cat_mappings['R1'].keys()),
            list(cat_mappings['R2'].keys()),
            list(cat_mappings['R3'].keys())
        ))

        # 🌟 关键修正：创建更精细的数值变量网格，增加结果多样性
        temp_grid = np.linspace(25, 250, 10)  # 从 10 增加到 20
        press_grid = np.linspace(0.1, 1.0, 5) # 从 5 增加到 10
        al_m_grid = np.linspace(25, 12500, 10) # 从 10 增加到 20
        time_grid = np.linspace(1, 60, 5)    # 从 5 增加到 10
        
        numerical_vars_combos = list(itertools.product(temp_grid, press_grid, al_m_grid, time_grid))

        # 构建虚拟输入 DataFrame
        virtual_inputs = []
        for discrete_combo in discrete_vars_combos:
            for numerical_combo in numerical_vars_combos:
                row = {
                    'M': cat_mappings['M'][discrete_combo[0]],
                    'R1': cat_mappings['R1'][discrete_combo[1]],
                    'R2': cat_mappings['R2'][discrete_combo[2]],
                    'R3': cat_mappings['R3'][discrete_combo[3]],
                    'T': numerical_combo[0],
                    'P': numerical_combo[1],
                    'Al/M': numerical_combo[2],
                    'Time': numerical_combo[3],
                    'Cat': 1.0, # 假设 Cat 和 Cocat 为常数
                    'Cocat': 2.0
                }
                virtual_inputs.append(row)

        virtual_df = pd.DataFrame(virtual_inputs, columns=get_input_variables())
        
        # 步骤 2: 预测虚拟数据集的性能
        virtual_df_with_preds = virtual_df.copy()
        for target in get_output_variables():
            # 这里的模型和标量需要从 load_models_and_data 函数传入
            scaler_X = scalers_X[target]
            model = models[target]
            
            scaled_input = scaler_X.transform(virtual_df)
            y_pred_scaled = model.predict(scaled_input)
            y_pred = scalers_y[target].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            virtual_df_with_preds[target] = y_pred

        st.success(f"已生成 {len(virtual_df_with_preds)} 个虚拟实验条件。")
        
        # 步骤 3: 寻找最接近目标值的条件（纯输出空间相似性搜索）
        
        combined_df_normalized = virtual_df_with_preds.copy()
        target_values = {}
        output_vars = get_output_variables()
        
        # 归一化输出变量 (CA, MW, MWD)
        for var in output_vars:
            mean = virtual_df_with_preds[var].mean()
            std = virtual_df_with_preds[var].std()
            
            # 🌟 关键：对标准差为0的情况进行处理，避免除以0
            if std == 0 or np.isnan(std):
                combined_df_normalized[var] = 0
                target_values[var] = 0
            else:
                combined_df_normalized[var] = (virtual_df_with_preds[var] - mean) / std

        # 归一化目标值
        target_values['CA'] = (target_ca - virtual_df_with_preds['CA'].mean()) / virtual_df_with_preds['CA'].std()
        target_values['MW'] = (target_mw - virtual_df_with_preds['MW'].mean()) / virtual_df_with_preds['MW'].std()
        target_values['MWD'] = (target_mwd - virtual_df_with_preds['MWD'].mean()) / virtual_df_with_preds['MWD'].std()
        
        # 4. 计算距离 (只针对归一化后的 CA, MW, MWD)
        normalized_data_matrix = combined_df_normalized[['CA', 'MW', 'MWD']].values
        target_vector = np.array([target_values['CA'], target_values['MW'], target_values['MWD']])
        
        distances = np.linalg.norm(normalized_data_matrix - target_vector, axis=1)
        virtual_df_with_preds['distance'] = distances
        
        # 5. 排序和提取结果
        closest_matches = virtual_df_with_preds.sort_values(by='distance').head(5)
        
        inverse_map = {col: {v: k for k, v in cat_mappings[col].items()} for col in ['M', 'R1', 'R2', 'R3']}

        st.success('找到以下5个最接近的潜在实验条件：')
        
        # 6. 展示结果
        for i, (index, row) in enumerate(closest_matches.iterrows()):
            st.markdown(f"**--- 第 {i+1} 个最接近的条件 ---**")
            
            # 准备要展示的 DataFrame
            result_df = row.drop(['CA', 'MW', 'MWD', 'distance']).to_frame().T
            
            # 逆向映射离散变量
            for col in ['R1', 'R2', 'R3', 'M']:
                if col in result_df.columns:
                    value = float(row[col])
                    # 使用 .get 方法避免 KeyError
                    ligand_name = inverse_map[col].get(value, f"未知配体({value})") 
                    result_df[col] = ligand_name
            
            # 格式化 T, P, Al/M, Time 等数值
            result_df['T'] = result_df['T'].round(1)
            result_df['P'] = result_df['P'].round(2)
            result_df['Al/M'] = result_df['Al/M'].round(0).astype(int)
            result_df['Time'] = result_df['Time'].round(1)
            
            st.dataframe(result_df)
            
            unit_map = {
                'CA': 'g·mmol⁻¹·h⁻¹',
                'MW': 'kg·mol⁻¹',
                'MWD': '(无单位)'
            }
            st.markdown(f"**该条件的预测性能：**")
            st.info(f"CA: {row['CA']:.2f} {unit_map['CA']} | MW: {row['MW']:.2f} {unit_map['MW']} | MWD: {row['MWD']:.2f} {unit_map['MWD']}")
            
            with st.expander(f"查看该条件的配体分子结构"):
                # 逆向映射离散变量的名称用于结构展示
                selected_ligand_names = {
                    'R1': inverse_map['R1'].get(float(row['R1']), "未知"),
                    'R2': inverse_map['R2'].get(float(row['R2']), "未知"),
                    'R3': inverse_map['R3'].get(float(row['R3']), "未知")
                }
                display_ligand_images(selected_ligand_names)
                
def main_model_performance(metrics):
    """模型性能展示页面"""
    st.title('模型性能评估')
    st.write('以下是每个预测模型在训练数据上的表现。')
    
    unit_map = {
        'CA': 'g·mmol⁻¹·h⁻¹',
        'MW': 'kg·mol⁻¹',
        'MWD': '(无单位)'
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header('CA')
        st.metric(label="R² Score", value=f"{metrics['CA']['R2']:.4f}")
        st.metric(label=f"RMSE ({unit_map['CA']})", value=f"{metrics['CA']['RMSE']:.2f}")
        st.metric(label=f"MAE ({unit_map['CA']})", value=f"{metrics['CA']['MAE']:.2f}")
    with col2:
        st.header('MW')
        st.metric(label="R² Score", value=f"{metrics['MW']['R2']:.4f}")
        st.metric(label=f"RMSE ({unit_map['MW']})", value=f"{metrics['MW']['RMSE']:.2f}")
        st.metric(label=f"MAE ({unit_map['MW']})", value=f"{metrics['MW']['MAE']:.2f}")
    with col3:
        st.header('MWD')
        st.metric(label="R² Score", value=f"{metrics['MWD']['R2']:.4f}")
        st.metric(label=f"RMSE ({unit_map['MWD']})", value=f"{metrics['MWD']['RMSE']:.2f}")
        st.metric(label=f"MAE ({unit_map['MWD']})", value=f"{metrics['MWD']['MAE']:.2f}")

def main_data_visualization(combined_df):
    """数据探索与可视化页面"""
    st.title('数据探索与可视化')
    st.write('通过选择变量，探索不同参数对催化剂性能的影响。')
    
        # 临时诊断：显示数据框，确认数据已加载
    st.write("诊断: combined_df 的数据类型")
    st.write(combined_df.dtypes)

    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者使用 'WenQuanYi Micro Hei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    categorical_vars = list(get_categorical_mappings().keys())

    all_vars = get_input_variables() + get_output_variables()
    
    st.sidebar.subheader('绘图设置')
    x_axis = st.sidebar.selectbox('选择 X 轴变量', all_vars, index=0)
    y_axis = st.sidebar.selectbox('选择 Y 轴变量', get_output_variables(), index=0)
    
    # ⚠️ 修复：修正变量类型判断逻辑
    is_x_categorical = x_axis in categorical_vars
    is_y_categorical = y_axis in categorical_vars # 对于输出变量，这应该始终为 False，因为它不是分类变量
    
    # 临时诊断：显示数据框，确认数据已加载
    st.dataframe(combined_df.head())

    fig, ax = plt.subplots(figsize=(10, 6))

    if not is_x_categorical and not is_y_categorical:
        # 散点图
        sns.scatterplot(data=combined_df, x=x_axis, y=y_axis, ax=ax)
        ax.set_title(f'{y_axis} vs. {x_axis} 散点图')
        st.pyplot(fig)
    elif is_x_categorical and not is_y_categorical:
        # 小提琴图/箱线图
        sns.violinplot(data=combined_df, x=x_axis, y=y_axis, ax=ax, inner="quartile")
        sns.stripplot(data=combined_df, x=x_axis, y=y_axis, ax=ax, color='black', alpha=0.5, jitter=True)
        ax.set_title(f'{y_axis} by {x_axis} 小提琴图')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif not is_x_categorical and is_y_categorical:
        st.warning("请选择一个数值变量作为 Y 轴。")
    else:
        st.warning("请选择至少一个数值变量进行绘图。")

def main():
    st.sidebar.title("功能选择")
    app_mode = st.sidebar.radio("请选择一个功能", ["正向预测", "寻找相似条件", "模型性能", "数据可视化"])

    models, scalers_X, scalers_y, combined_df, metrics = load_models_and_data()

    if app_mode == "正向预测":
        main_forward_prediction(models, scalers_X, scalers_y, metrics)
    elif app_mode == "寻找相似条件":
        # 传递必要的参数给新的 inverse_search 函数
        main_inverse_search(models, scalers_X, scalers_y)
    elif app_mode == "模型性能":
        main_model_performance(metrics)
    elif app_mode == "数据可视化":
        main_data_visualization(combined_df)


# 最终修复后的主入口点
if __name__ == '__main__':

    main() 







