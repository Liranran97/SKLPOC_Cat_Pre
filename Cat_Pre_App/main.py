import os
import joblib
from utils import load_data, get_input_variables, get_output_variables, get_categorical_mappings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def train_and_save_models():
    """
    加载数据，为每个输出变量训练并保存一个随机森林模型。
    """
    
    # 定义模型保存路径
    model_dir = 'Models/Trained'
    os.makedirs(model_dir, exist_ok=True)
    
    data = load_data()
    
    input_vars = get_input_variables()
    
    # 为每个目标变量训练和保存模型
    for target_name, target_data in data.items():
        print(f"训练 {target_name} 模型...")
        
        # 合并训练和测试数据以进行统一的标准化处理
        X_train_val = pd.concat([target_data['train_x'], target_data['test_x']])
        y_train_val = pd.concat([target_data['train_y'], target_data['test_y']])

        # 实例化标准化器
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # 对特征和目标变量进行标准化
        X_scaled = scaler_X.fit_transform(X_train_val[input_vars])
        y_scaled = scaler_y.fit_transform(y_train_val.values.reshape(-1, 1))

        # 重新分割训练集和测试集（此处为演示，实际应用中应在标准化前分割）
        # 这里仅为确保模型能处理所有数据范围
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
        
        # 使用随机森林回归器
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.ravel())
        
        # 保存模型和标准化器
        model_path = os.path.join(model_dir, f'model_{target_name}.joblib')
        scaler_X_path = os.path.join(model_dir, f'scaler_X_{target_name}.joblib')
        scaler_y_path = os.path.join(model_dir, f'scaler_y_{target_name}.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(scaler_X, scaler_X_path)
        joblib.dump(scaler_y, scaler_y_path)
        
        print(f"{target_name} 模型训练完成并已保存到 {model_dir}")

    print("\n所有模型训练和保存完毕。")

if __name__ == '__main__':
    train_and_save_models()