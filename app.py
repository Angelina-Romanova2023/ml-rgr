import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
from PIL import Image

# ======================== Конфигурация страницы ========================
st.set_page_config(page_title="ML Diamonds Dashboard", layout="wide")

# ======================== Пути ========================
# Получаем текущую директорию скрипта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Папка для моделей
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

# Папка для данных
data_dir = os.path.join(BASE_DIR, "upload")
os.makedirs(data_dir, exist_ok=True)

# Полные пути к файлам
scaler_path = os.path.join(models_dir, "scaler.pkl")
dataset_path = os.path.join(data_dir, "diamonds_processed.csv")

# ======================== Загрузка моделей и скейлера ========================
@st.cache_resource
def load_models():
    models = {}
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Файл скейлера не найден: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    for i in range(1, 7):
        model_file = os.path.join(models_dir, f"ml{i}.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Файл модели не найден: {model_file}")
        with open(model_file, 'rb') as f:
            models[f"ML{i}"] = pickle.load(f)

    return models, scaler

# ======================== Загрузка датасета ========================
@st.cache_data
def load_dataset():
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Файл с данными не найден: {dataset_path}")
    df = pd.read_csv(dataset_path, sep=';')
    return df

# ======================== Функция для загрузки фото ========================
def load_developer_photo():
    try:
        # Пытаемся загрузить фото из локального файла
        photo_path = os.path.join(BASE_DIR, "developer_photo.jpg")
        if os.path.exists(photo_path):
            image = Image.open(photo_path)
            return image
    except Exception as e:
        st.error(f"Ошибка загрузки фото: {e}")
        return None
    
# ======================== Загрузка ресурсов ========================
models, scaler = load_models()
df = load_dataset()

# Боковая панель для навигации
st.sidebar.title("📊 Навигация")
page = st.sidebar.radio("Выберите страницу:", ["Информация о разработчике", "Информация о датасете", "Визуализация данных", "Предсказание модели"])

# ==================== PAGE 1: ИНФОРМАЦИЯ О РАЗРАБОТЧИКЕ ====================
if page == "Информация о разработчике":
    st.title("👨‍💼 Информация о разработчике")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Фото")
        # Загружаем фото
        photo = load_developer_photo()
        if photo:
            st.image(photo, caption="Романова Ангелина")
        else:
            # Если фото нет, показываем заглушку с эмодзи
            st.markdown("""
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h1 style='font-size: 100px; margin: 0;'>👨‍💻</h1>
                <p style='font-size: 18px; margin-top: 10px;'>Фото разработчика</p>
                <p style='font-size: 14px; color: #666;'>Поместите файл developer_photo.jpg в папку проекта</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### Данные разработчика
        - **ФИО:** Романова Ангелина
        - **Группа:** Фит-231
        - **Университет:** ОмГТУ (Омский государственный технический университет)
        - **Предмет:** Машинное обучение и большие данные
        - **Тема проекта:** Разработка веб-приложения (Dashboard) для инференса моделей ML и анализа данных
        
        ### Цель проекта
        Данный проект демонстрирует практическое применение моделей машинного обучения для задач регрессии,
        а именно прогнозирования цен на бриллианты на основе их характеристик.
        """)
    
    st.markdown("---")
    st.markdown("""
    ### Используемые технологии
    - **Python**
    - **Streamlit** - фреймворк для веб-приложений
    - **Scikit-learn** - библиотека машинного обучения
    - **Pandas** - обработка данных
    - **Matplotlib & Seaborn** - визуализация данных
    """)

# ==================== PAGE 2: ИНФОРМАЦИЯ О ДАТАСЕТЕ ====================
elif page == "Информация о датасете":
    st.title("📈 Информация о датасете")
    
    st.markdown("### Обзор датасета")
    st.write(f"**Всего записей:** {len(df):,}")
    st.write(f"**Всего признаков:** {df.shape[1]}")
    
    st.markdown("### Статистика датасета")
    st.dataframe(df.describe().round(2))
    
    st.markdown("### Информация о признаках")
    feature_info = pd.DataFrame({
        'Признак': df.columns,
        'Тип данных': df.dtypes,
        'Пропущенные значения': df.isnull().sum(),
        'Уникальные значения': df.nunique()
    })
    st.dataframe(feature_info)
    
    st.markdown("### Целевая переменная (Цена)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Мин. цена", f"${df['price'].min():,.0f}")
    with col2:
        st.metric("Средняя цена", f"${df['price'].mean():,.0f}")
    with col3:
        st.metric("Макс. цена", f"${df['price'].max():,.0f}")
    
    st.markdown("### Предобработка данных")
    st.info("""
    ✓ Обработаны пропущенные значения
    ✓ Категориальные признаки закодированы
    ✓ Числовые признаки масштабированы с помощью StandardScaler
    ✓ Данные разделены: 80% обучение, 20% тестирование
    """)

# ==================== PAGE 3: ВИЗУАЛИЗАЦИЯ ДАННЫХ ====================
elif page == "Визуализация данных":
    st.title("📊 Визуализация данных")
    
    st.markdown("### Галерея визуализаций")
    
    # Получаем числовые колонки для визуализации
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Удаляем 'price' из опций для оси X
    x_cols = [col for col in numeric_cols if col != 'price']
    
    col1, col2 = st.columns(2)
    
    # Визуализация 1: Распределение цен
    with col1:
        st.markdown("#### 1. Распределение цен")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['price'], bins=50, color='skyblue', edgecolor='black')
        ax.set_xlabel('Цена ($)')
        ax.set_ylabel('Частота')
        ax.set_title('Распределение цен на бриллианты')
        st.pyplot(fig)
    
    # Визуализация 2: Точечный график
    with col2:
        st.markdown("#### 2. Цена vs Первый признак")
        if len(x_cols) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(df[x_cols[0]], df['price'], alpha=0.5, color='green')
            ax.set_xlabel(x_cols[0])
            ax.set_ylabel('Цена ($)')
            ax.set_title(f'Цена vs {x_cols[0]}')
            st.pyplot(fig)
    
    col3, col4 = st.columns(2)
    
    # Визуализация 3: Ящик с усами
    with col3:
        st.markdown("#### 3. Ящик с усами (Box Plot)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(df['price'], vert=True)
        ax.set_ylabel('Цена ($)')
        ax.set_title('Box Plot цен')
        st.pyplot(fig)
    
    # Визуализация 4: Тепловая карта корреляций
    with col4:
        st.markdown("#### 4. Корреляция признаков")
        fig, ax = plt.subplots(figsize=(8, 5))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'label': 'Корреляция'})
        ax.set_title('Матрица корреляции признаков')
        st.pyplot(fig)

# ==================== PAGE 4: ПРЕДСКАЗАНИЕ МОДЕЛИ ====================
elif page == "Предсказание модели":
    st.title("🤖 Предсказание модели")
    
    st.markdown("### Доступные модели")
    model_info = {
        'ML1': {'name': 'Линейная регрессия', 'r2': 0.7498},
        'ML2': {'name': 'Градиентный бустинг', 'r2': 0.9305},
        'ML3': {'name': 'HistGradientBoosting', 'r2': 0.9547},
        'ML4': {'name': 'Случайный лес', 'r2': 0.9526},
        'ML5': {'name': 'Стекинг ансамбль', 'r2': 0.9477},
        'ML6': {'name': 'Нейронная сеть MLP', 'r2': 0.9281}
    }
    
    # Отображение производительности моделей
    col1, col2, col3 = st.columns(3)
    models_list = list(model_info.keys())
    
    for idx, (model_key, info) in enumerate(model_info.items()):
        if idx % 3 == 0:
            col = col1
        elif idx % 3 == 1:
            col = col2
        else:
            col = col3
        
        with col:
            st.metric(f"{model_key}: {info['name']}", f"R² = {info['r2']:.4f}")
    
    st.markdown("---")
    st.markdown("### Сделать предсказание")
    
    # Выбор метода ввода
    input_method = st.radio("Выберите способ ввода:", ["Ручной ввод", "Загрузить CSV"])
    
    if input_method == "Ручной ввод":
        st.markdown("#### Введите значения признаков")
        
        # Получаем названия признаков (исключая 'price')
        feature_names = [col for col in df.columns if col != 'price']
        
        # Создаем поля для ввода
        input_values = {}
        cols = st.columns(3)
        
        for idx, feature in enumerate(feature_names):
            col_idx = idx % 3
            with cols[col_idx]:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                input_values[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100
                )
        
        # Подготовка данных для предсказания
        if st.button("🔮 Предсказать цену"):
            try:
                # Создаем DataFrame с введенными значениями
                input_df = pd.DataFrame([input_values])
                
                # Масштабируем входные данные
                input_scaled = scaler.transform(input_df)
                
                # Делаем предсказания всеми моделями
                predictions = {}
                for model_name, model in models.items():
                    pred = model.predict(input_scaled)[0]
                    predictions[model_name] = pred
                
                st.markdown("---")
                st.markdown("### Результаты предсказания")
                
                # Отображаем предсказания
                col1, col2, col3 = st.columns(3)
                for idx, (model_name, pred_price) in enumerate(predictions.items()):
                    col_idx = idx % 3
                    if col_idx == 0:
                        col = col1
                    elif col_idx == 1:
                        col = col2
                    else:
                        col = col3
                    
                    with col:
                        st.info(f"**{model_name}**\n${pred_price:,.2f}")
                
                # Среднее предсказание
                avg_price = np.mean(list(predictions.values()))
                st.success(f"### Среднее предсказание: ${avg_price:,.2f}")
                
            except Exception as e:
                st.error(f"Ошибка во время предсказания: {str(e)}")
    
    elif input_method == "Загрузить CSV":
        st.markdown("#### Загрузите CSV файл")
        uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
        
        if uploaded_file is not None:
            try:
                # Читаем загруженный файл
                input_df = pd.read_csv(uploaded_file, sep=';')
                
                # Отображаем загруженные данные
                st.markdown("**Предпросмотр загруженных данных:**")
                st.dataframe(input_df.head())
                
                # Делаем предсказания
                if st.button("🔮 Предсказать для всех строк"):
                    input_scaled = scaler.transform(input_df)
                    
                    results_df = input_df.copy()
                    
                    for model_name, model in models.items():
                        predictions = model.predict(input_scaled)
                        results_df[f'{model_name}_Предсказание'] = predictions
                    
                    # Вычисляем среднее предсказание
                    pred_cols = [col for col in results_df.columns if 'Предсказание' in col]
                    results_df['Средняя_цена'] = results_df[pred_cols].mean(axis=1)
                    
                    st.markdown("**Результаты предсказания:**")
                    st.dataframe(results_df[['Средняя_цена'] + pred_cols].round(2))
                    
                    # Скачивание результатов
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Скачать результаты",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><small>ML Diamonds Dashboard | ОмГТУ | 2026</small></p>
</div>
""", unsafe_allow_html=True)