import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ==============================================================================
# Настройка страницы и стилей
# ==============================================================================
st.set_page_config(layout="wide", page_title="Лабораторная по нечеткой логике")

st.title("🔬 Интерактивная лабораторная работа: Нечеткая логика")
st.write("""
Это веб-приложение демонстрирует основные концепции нечеткой логики, описанные в лабораторной работе.
Используйте боковую панель для навигации по разделам.
""")

# ==============================================================================
# Боковая панель для навигации
# ==============================================================================
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите раздел лабораторной работы:",
                        ["Введение",
                         "Часть 1: Функции принадлежности",
                         "Часть 2: Операции над нечеткими множествами",
                         "Часть 3: Нечеткая аппроксимирующая система"])

# ==============================================================================
# Раздел: Введение
# ==============================================================================
if page == "Введение":
    st.header("Цель работы")
    st.markdown("""
    - Изучить основные определения теории нечётких множеств.
    - Ознакомиться со способами задания функций принадлежности и научиться их строить.
    - Изучить принципы построения нечётких аппроксимирующих систем.

    В этой интерактивной среде вы можете изменять параметры и в реальном времени видеть, как меняются
    результаты, что помогает лучше понять материал.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Fuzzy_logic_temperature.svg/500px-Fuzzy_logic_temperature.svg.png",
             caption="Пример функций принадлежности для переменной 'Температура'", width=500)

# ==============================================================================
# Раздел 1: Функции принадлежности
# ==============================================================================
elif page == "Часть 1: Функции принадлежности":
    st.header("Часть 1: Построение функций принадлежности (ФП)")
    st.markdown("Выберите тип функции и настройте её параметры с помощью слайдеров.")

    func_type = st.selectbox("Выберите тип ФП:",
                             ["Треугольная (trimf)", "Трапециевидная (trapmf)",
                              "Гауссова (gaussmf)", "Обобщенный колокол (gbellmf)",
                              "Сигмоидная (sigmf)", "Z-образная (zmf)", "S-образная (smf)"])

    fig, ax = plt.subplots()
    # Универсальный диапазон для всех графиков
    x = np.arange(0, 25.1, 0.1)

    if func_type == "Треугольная (trimf)":
        st.subheader("Треугольная ФП: `fuzz.trimf(x, [a, b, c])`")
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.slider("Параметр 'a' (левое основание):", 0.0, 25.0, 5.0)
        with col2:
            b = st.slider("Параметр 'b' (вершина):", 0.0, 25.0, 10.0)
        with col3:
            c = st.slider("Параметр 'c' (правое основание):", 0.0, 25.0, 15.0)
        
        params = [a, b, c]
        # Проверка корректности параметров
        if not (a <= b <= c):
            st.error("Ошибка: Параметры должны удовлетворять условию a <= b <= c.")
            y = np.zeros_like(x) # Рисуем пустой график в случае ошибки
        else:
            y = fuzz.trimf(x, params)
        
        ax.plot(x, y, 'b', linewidth=2)
        ax.set_title(f"trimf с параметрами {params}")

    elif func_type == "Трапециевидная (trapmf)":
        st.subheader("Трапециевидная ФП: `fuzz.trapmf(x, [a, b, c, d])`")
        base_params = st.slider("Основание [a, d]:", 0.0, 25.0, (3.0, 22.0))
        top_params = st.slider("Вершина [b, c]:", 0.0, 25.0, (8.0, 15.0))
        
        a, d = base_params
        b, c = top_params

        params = [a, b, c, d]
        # Проверка корректности параметров
        if not (a <= b <= c <= d):
            st.error("Ошибка: Параметры должны удовлетворять условию a <= b <= c <= d.")
            y = np.zeros_like(x) # Рисуем пустой график в случае ошибки
        else:
            y = fuzz.trapmf(x, params)
            
        ax.plot(x, y, 'g', linewidth=2)
        ax.set_title(f"trapmf с параметрами {params}")

    elif func_type == "Гауссова (gaussmf)":
        st.subheader("Гауссова ФП: `fuzz.gaussmf(x, mean, sigma)`")
        mean = st.slider("Центр (mean):", 0.0, 25.0, 12.0)
        sigma = st.slider("Ширина (sigma):", 0.1, 10.0, 2.0)
        y = fuzz.gaussmf(x, mean, sigma)
        ax.plot(x, y, 'r', linewidth=2)
        ax.set_title(f"gaussmf с центром={mean}, шириной={sigma}")

    elif func_type == "Обобщенный колокол (gbellmf)":
        st.subheader("ФП 'Обобщенный колокол': `fuzz.gbellmf(x, a, b, c)`")
        a = st.slider("Ширина (a):", 0.1, 10.0, 2.0)
        b = st.slider("Наклон (b):", 0.1, 10.0, 4.0)
        c = st.slider("Центр (c):", 0.0, 25.0, 12.0)
        y = fuzz.gbellmf(x, a, b, c)
        ax.plot(x, y, 'm', linewidth=2)
        ax.set_title(f"gbellmf с a={a}, b={b}, c={c}")

    elif func_type == "Сигмоидная (sigmf)":
        st.subheader("Сигмоидная ФП: `fuzz.sigmf(x, center, slope)`")
        center = st.slider("Центр (center):", 0.0, 25.0, 12.0)
        slope = st.slider("Наклон (slope):", -10.0, 10.0, 1.0)
        y = fuzz.sigmf(x, center, slope)
        ax.plot(x, y, 'c', linewidth=2)
        ax.set_title(f"sigmf с центром={center}, наклоном={slope}")

    elif func_type == "Z-образная (zmf)":
        st.subheader("Z-образная ФП: `fuzz.zmf(x, a, b)`")
        params = st.slider("Точки спада [a, b]:", 0.0, 25.0, (5.0, 15.0))
        y = fuzz.zmf(x, params[0], params[1])
        ax.plot(x, y, 'y', linewidth=2)
        ax.set_title(f"zmf с параметрами {params}")
        
    elif func_type == "S-образная (smf)":
        st.subheader("S-образная ФП: `fuzz.smf(x, a, b)`")
        params = st.slider("Точки подъема [a, b]:", 0.0, 25.0, (5.0, 15.0))
        y = fuzz.smf(x, params[0], params[1])
        ax.plot(x, y, 'k', linewidth=2)
        ax.set_title(f"smf с параметрами {params}")

    ax.grid(True)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 25)
    st.pyplot(fig)


# ==============================================================================
# Раздел 2: Операции над нечеткими множествами
# ==============================================================================
elif page == "Часть 2: Операции над нечеткими множествами":
    st.header("Часть 2: Операции над нечеткими множествами")
    st.markdown("Настройте два нечетких множества (A и B) и посмотрите результаты операций над ними.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Множество A (Гауссова ФП)")
        mean_a = st.slider("Центр A:", 0.0, 20.0, 12.0)
        sigma_a = st.slider("Ширина A:", 0.1, 10.0, 2.0)

    with col2:
        st.subheader("Множество B (Гауссова ФП)")
        mean_b = st.slider("Центр B:", 0.0, 20.0, 8.0)
        sigma_b = st.slider("Ширина B:", 0.1, 10.0, 3.0)

    x = np.arange(0, 20.1, 0.1)
    A = fuzz.gaussmf(x, mean_a, sigma_a)
    B = fuzz.gaussmf(x, mean_b, sigma_b)

    # Операции
    intersect_min = np.fmin(A, B)
    union_max = np.fmax(A, B)
    complement_A = 1 - A
    
    # Визуализация
    st.subheader("Результаты операций")
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Исходные множества
    ax0.plot(x, A, 'b', linewidth=1.5, label='Множество A')
    ax0.plot(x, B, 'g', linewidth=1.5, label='Множество B')
    ax0.set_title("Исходные множества")
    ax0.legend()
    ax0.grid(True)

    # Пересечение (AND)
    ax1.fill_between(x, 0, intersect_min, facecolor='orange', alpha=0.7)
    ax1.plot(x, A, 'b:', linewidth=1)
    ax1.plot(x, B, 'g:', linewidth=1)
    ax1.set_title("Пересечение (min(A, B))")
    ax1.grid(True)

    # Объединение (OR)
    ax2.fill_between(x, 0, union_max, facecolor='cyan', alpha=0.7)
    ax2.plot(x, A, 'b:', linewidth=1)
    ax2.plot(x, B, 'g:', linewidth=1)
    ax2.set_title("Объединение (max(A, B))")
    ax2.grid(True)
    
    # Дополнение (NOT)
    ax3.plot(x, A, 'b:', linewidth=1, label='Множество A')
    ax3.fill_between(x, 0, complement_A, facecolor='red', alpha=0.7, label='Дополнение A')
    ax3.set_title("Дополнение (1 - A)")
    ax3.legend()
    ax3.grid(True)

    for ax_ in fig.get_axes():
        ax_.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    st.pyplot(fig)


# ==============================================================================
# Раздел 3: Нечеткая аппроксимирующая система
# ==============================================================================
elif page == "Часть 3: Нечеткая аппроксимирующая система":
    st.header("Часть 3: Аппроксимация функции `y = x^2`")
    st.markdown("""
    Здесь мы строим нечеткую систему (FIS) для аппроксимации функции `y = x^2` на интервале `[-1, 1]`.
    Система использует 5 правил, основанных на данных из таблицы 1 лабораторной работы.
    """)

    # --- Функции для кэширования ---
    @st.cache_data
    def create_fis(sigma):
        x_input = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), 'x')
        y_output = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'y')
        
        x_input['bn'] = fuzz.gaussmf(x_input.universe, -1.0, sigma)
        x_input['n']  = fuzz.gaussmf(x_input.universe, -0.6, sigma)
        x_input['z']  = fuzz.gaussmf(x_input.universe,  0.0, sigma)
        x_input['p']  = fuzz.gaussmf(x_input.universe,  0.4, sigma)
        x_input['pb'] = fuzz.gaussmf(x_input.universe,  1.0, sigma)

        y_output['level_1']    = fuzz.trimf(y_output.universe, [1.0, 1.0, 1.0])
        y_output['level_036']  = fuzz.trimf(y_output.universe, [0.36, 0.36, 0.36])
        y_output['level_0']    = fuzz.trimf(y_output.universe, [0, 0, 0])
        y_output['level_016']  = fuzz.trimf(y_output.universe, [0.16, 0.16, 0.16])
        
        rule1 = ctrl.Rule(x_input['bn'], y_output['level_1'])
        rule2 = ctrl.Rule(x_input['n'],  y_output['level_036'])
        rule3 = ctrl.Rule(x_input['z'],  y_output['level_0'])
        rule4 = ctrl.Rule(x_input['p'],  y_output['level_016'])
        rule5 = ctrl.Rule(x_input['pb'], y_output['level_1'])
        
        fis_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        return fis_control, x_input

    # --- Интерфейс ---
    st.sidebar.subheader("Параметры системы")
    sigma = st.sidebar.slider("Ширина (sigma) для всех ФП входа:", 0.05, 0.5, 0.15)
    
    fis_control, x_input = create_fis(sigma)
    fis_simulation = ctrl.ControlSystemSimulation(fis_control)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Функции принадлежности для входа 'x'")
        fig_mf, ax_mf = plt.subplots()
        for term_name, term_obj in x_input.terms.items():
            ax_mf.plot(x_input.universe, term_obj.mf, label=term_name)
        ax_mf.set_title(f"ФП для входа при sigma={sigma:.2f}")
        ax_mf.grid(True)
        ax_mf.legend()
        st.pyplot(fig_mf)

        st.subheader("2. Правила системы")
        st.code("""
        1. IF (x IS bn) THEN (y IS 1.0)
        2. IF (x IS n)  THEN (y IS 0.36)
        3. IF (x IS z)  THEN (y IS 0.0)
        4. IF (x IS p)  THEN (y IS 0.16)
        5. IF (x IS pb) THEN (y IS 1.0)
        """, language="none")

    with col2:
        st.subheader("3. Тестирование системы (аналог Rule Viewer)")
        input_x = st.slider("Выберите входное значение 'x':", -1.0, 1.0, -0.09, 0.01) # Установил ваше значение по умолчанию
        fis_simulation.input['x'] = input_x
        fis_simulation.compute()
        output_y = fis_simulation.output['y']

        st.metric(label=f"Выход 'y' для x = {input_x}", value=f"{output_y:.4f}")

        # ======== НАЧАЛО ИЗМЕНЕНИЙ ========
        # Создаем график для визуализации активации правил вручную
        fig_sim, ax_sim = plt.subplots()

        # Рисуем все функции принадлежности
        for term_name, term_obj in x_input.terms.items():
            ax_sim.plot(x_input.universe, term_obj.mf, label=term_name)

            # Вычисляем степень активации для текущего входа
            activation_level = fuzz.interp_membership(x_input.universe, term_obj.mf, input_x)
            
            # Создаем "срезанную" версию ФП для заливки
            capped_mf = np.fmin(activation_level, term_obj.mf)
            ax_sim.fill_between(x_input.universe, 0, capped_mf, alpha=0.4)

        # Рисуем вертикальную линию, показывающую текущее значение входа
        ax_sim.axvline(x=input_x, color='k', linestyle='--', linewidth=2, label=f'Вход x={input_x:.2f}')
        
        ax_sim.set_title(f"Активация ФП для x = {input_x:.2f}")
        ax_sim.grid(True)
        ax_sim.legend()
        st.pyplot(fig_sim)
        # ======== КОНЕЦ ИЗМЕНЕНИЙ ========


    st.subheader("4. Поверхность отклика (результат аппроксимации)")
    
    @st.cache_data
    def calculate_response_curve(_fis_control):
        x_values = np.linspace(-1, 1, 101)
        y_values = np.zeros_like(x_values)
        temp_sim = ctrl.ControlSystemSimulation(_fis_control)
        
        for i, x_val in enumerate(x_values):
            temp_sim.input['x'] = x_val
            temp_sim.compute()
            y_values[i] = temp_sim.output['y']
        return x_values, y_values

    x_vals, y_vals = calculate_response_curve(fis_control)
    y_ideal = x_vals**2

    fig_resp, ax_resp = plt.subplots(figsize=(10, 6))
    ax_resp.plot(x_vals, y_vals, 'b', linewidth=2, label='Выход нечеткой системы')
    ax_resp.plot(x_vals, y_ideal, 'r--', linewidth=2, label='Идеальная функция y = x^2')
    ax_resp.set_title('Сравнение результата аппроксимации с идеальной функцией')
    ax_resp.set_xlabel('Вход: x')
    ax_resp.set_ylabel('Выход: y')
    ax_resp.grid(True)
    ax_resp.legend()
    st.pyplot(fig_resp)
    st.info("Попробуйте изменить параметр `sigma` на боковой панели и посмотрите, как это повлияет на точность аппроксимации.")