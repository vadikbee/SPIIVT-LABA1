import tkinter as tk
from tkinter import ttk
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================================================================
# Класс основного приложения
# ==============================================================================
class FuzzyLogicLabApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Лабораторная работа: Нечеткая логика")

        # Настройка вкладок
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")

        # Создание вкладок для каждой части
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text="Часть 1: Функции принадлежности")
        self.notebook.add(self.tab2, text="Часть 2: Операции над множествами")
        self.notebook.add(self.tab3, text="Часть 3: Аппроксимирующая система")
        
        # Заполнение каждой вкладки содержимым
        self.create_part1_ui()
        self.create_part2_ui()
        self.create_part3_ui()

    # ==============================================================================
    # ЧАСТЬ 1: ФУНКЦИИ ПРИНАДЛЕЖНОСТИ
    # ==============================================================================
    def create_part1_ui(self):
        controls_frame = ttk.LabelFrame(self.tab1, text="Управление", padding=(10, 5))
        controls_frame.pack(side="left", fill="y", padx=10, pady=10)

        plot_frame = ttk.Frame(self.tab1)
        plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ttk.Label(controls_frame, text="Выберите тип ФП:").pack(pady=(0, 5))
        self.func_type_var = tk.StringVar(value="Треугольная (trimf)")
        func_types = ["Треугольная (trimf)", "Трапециевидная (trapmf)", "Гауссова (gaussmf)",
                      "Обобщенный колокол (gbellmf)", "Сигмоидная (sigmf)", "Z-образная (zmf)",
                      "S-образная (smf)"]
        func_menu = ttk.Combobox(controls_frame, textvariable=self.func_type_var, values=func_types, state='readonly', width=25)
        func_menu.pack(pady=(0, 15))
        func_menu.bind("<<ComboboxSelected>>", self.update_part1_controls)
        
        self.p1_params_frame = ttk.Frame(controls_frame)
        self.p1_params_frame.pack(fill='x', expand=True)

        self.p1_error_label_var = tk.StringVar()
        ttk.Label(controls_frame, textvariable=self.p1_error_label_var, foreground="red").pack(pady=10)


        self.fig1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plot_frame)
        self.canvas1.get_tk_widget().pack(fill="both", expand=True)

        self.p1_controls = {}
        self.update_part1_controls()

    def update_part1_controls(self, event=None):
        for widget in self.p1_params_frame.winfo_children():
            widget.destroy()

        self.p1_controls = {}
        func_type = self.func_type_var.get()
        
        if "Треугольная" in func_type:
            self.p1_controls['a'] = self.create_slider("a (левое основание):", 0.0, 25.0, 5.0)
            self.p1_controls['b'] = self.create_slider("b (вершина):", 0.0, 25.0, 10.0)
            self.p1_controls['c'] = self.create_slider("c (правое основание):", 0.0, 25.0, 15.0)
        elif "Трапециевидная" in func_type:
            self.p1_controls['a'] = self.create_slider("a (левое основание):", 0.0, 25.0, 3.0)
            self.p1_controls['b'] = self.create_slider("b (левая вершина):", 0.0, 25.0, 8.0)
            self.p1_controls['c'] = self.create_slider("c (правая вершина):", 0.0, 25.0, 15.0)
            self.p1_controls['d'] = self.create_slider("d (правое основание):", 0.0, 25.0, 22.0)
        elif "Гауссова" in func_type:
            self.p1_controls['mean'] = self.create_slider("Центр (mean):", 0.0, 25.0, 12.0)
            self.p1_controls['sigma'] = self.create_slider("Ширина (sigma):", 0.1, 10.0, 2.0)
        elif "Обобщенный колокол" in func_type:
            self.p1_controls['a'] = self.create_slider("Ширина (a):", 0.1, 10.0, 2.0)
            self.p1_controls['b'] = self.create_slider("Наклон (b):", 0.1, 10.0, 4.0)
            self.p1_controls['c'] = self.create_slider("Центр (c):", 0.0, 25.0, 12.0)
        elif "Сигмоидная" in func_type:
            self.p1_controls['center'] = self.create_slider("Центр (center):", 0.0, 25.0, 12.0)
            self.p1_controls['slope'] = self.create_slider("Наклон (slope):", -10.0, 10.0, 1.0)
        elif "Z-образная" in func_type:
            self.p1_controls['a'] = self.create_slider("Точка спада a:", 0.0, 25.0, 5.0)
            self.p1_controls['b'] = self.create_slider("Точка спада b:", 0.0, 25.0, 15.0)
        elif "S-образная" in func_type:
            self.p1_controls['a'] = self.create_slider("Точка подъема a:", 0.0, 25.0, 5.0)
            self.p1_controls['b'] = self.create_slider("Точка подъема b:", 0.0, 25.0, 15.0)
            
        self.update_part1_plot()

    def create_slider(self, text, from_, to, default_val):
        frame = ttk.Frame(self.p1_params_frame)
        ttk.Label(frame, text=text).pack(anchor='w')
        var = tk.DoubleVar(value=default_val)
        slider = ttk.Scale(frame, from_=from_, to=to, orient="horizontal", variable=var, command=self.update_part1_plot)
        slider.pack(fill='x')
        frame.pack(pady=5, fill='x')
        return var

    def update_part1_plot(self, event=None):
        self.p1_error_label_var.set("") 
        self.ax1.clear()
        x = np.arange(0, 25.1, 0.1)
        func_type = self.func_type_var.get()
        vals = {name: var.get() for name, var in self.p1_controls.items()}

        y = np.zeros_like(x)
        title = ""

        try:
            if "Треугольная" in func_type:
                a, b, c = vals['a'], vals['b'], vals['c']
                if not (a <= b <= c): self.p1_error_label_var.set("Ошибка: a <= b <= c")
                else: y = fuzz.trimf(x, [a, b, c])
                title = f"trimf(x, [{a:.1f}, {b:.1f}, {c:.1f}])"
            elif "Трапециевидная" in func_type:
                a,b,c,d = vals['a'], vals['b'], vals['c'], vals['d']
                if not (a <= b <= c <= d): self.p1_error_label_var.set("Ошибка: a <= b <= c <= d")
                else: y = fuzz.trapmf(x, [a, b, c, d])
                title = f"trapmf(x, [{a:.1f}, {b:.1f}, {c:.1f}, {d:.1f}])"
            elif "Гауссова" in func_type:
                y = fuzz.gaussmf(x, vals['mean'], vals['sigma'])
                title = f"gaussmf(x, mean={vals['mean']:.1f}, sigma={vals['sigma']:.1f})"
            elif "Обобщенный колокол" in func_type:
                y = fuzz.gbellmf(x, vals['a'], vals['b'], vals['c'])
                title = f"gbellmf(x, a={vals['a']:.1f}, b={vals['b']:.1f}, c={vals['c']:.1f})"
            elif "Сигмоидная" in func_type:
                y = fuzz.sigmf(x, vals['center'], vals['slope'])
                title = f"sigmf(x, center={vals['center']:.1f}, slope={vals['slope']:.1f})"
            elif "Z-образная" in func_type:
                a, b = vals['a'], vals['b']
                if not(a <= b): self.p1_error_label_var.set("Ошибка: a <= b")
                else: y = fuzz.zmf(x, a, b)
                title = f"zmf(x, a={a:.1f}, b={b:.1f})"
            elif "S-образная" in func_type:
                a, b = vals['a'], vals['b']
                if not(a <= b): self.p1_error_label_var.set("Ошибка: a <= b")
                else: y = fuzz.smf(x, a, b)
                title = f"smf(x, a={a:.1f}, b={b:.1f})"
        except Exception as e:
            self.p1_error_label_var.set(f"Ошибка: {e}")

        self.ax1.plot(x, y, linewidth=2)
        self.ax1.set_title(title)
        self.ax1.grid(True)
        self.ax1.set_ylim(-0.05, 1.05)
        self.ax1.set_xlim(0, 25)
        self.canvas1.draw()

    # ==============================================================================
    # ЧАСТЬ 2: ОПЕРАЦИИ НАД НЕЧЕТКИМИ МНОЖЕСТВАМИ
    # ==============================================================================
    def create_part2_ui(self):
        controls_frame = ttk.LabelFrame(self.tab2, text="Управление", padding=(10, 5))
        controls_frame.pack(side="top", fill="x", padx=10, pady=10)
        
        frame_a = ttk.LabelFrame(controls_frame, text="Множество A (Гауссова ФП)")
        frame_a.pack(side="left", padx=10, expand=True, fill="x")
        frame_b = ttk.LabelFrame(controls_frame, text="Множество B (Гауссова ФП)")
        frame_b.pack(side="right", padx=10, expand=True, fill="x")

        self.p2_mean_a = tk.DoubleVar(value=12.0); self.p2_sigma_a = tk.DoubleVar(value=2.0)
        self.p2_mean_b = tk.DoubleVar(value=8.0);  self.p2_sigma_b = tk.DoubleVar(value=3.0)
        
        ttk.Label(frame_a, text="Центр A:").pack()
        ttk.Scale(frame_a, from_=0.0, to=20.0, orient="horizontal", variable=self.p2_mean_a, command=self.update_part2_plot).pack(fill='x', expand=True, pady=5)
        ttk.Label(frame_a, text="Ширина A:").pack()
        ttk.Scale(frame_a, from_=0.1, to=10.0, orient="horizontal", variable=self.p2_sigma_a, command=self.update_part2_plot).pack(fill='x', expand=True, pady=5)
        ttk.Label(frame_b, text="Центр B:").pack()
        ttk.Scale(frame_b, from_=0.0, to=20.0, orient="horizontal", variable=self.p2_mean_b, command=self.update_part2_plot).pack(fill='x', expand=True, pady=5)
        ttk.Label(frame_b, text="Ширина B:").pack()
        ttk.Scale(frame_b, from_=0.1, to=10.0, orient="horizontal", variable=self.p2_sigma_b, command=self.update_part2_plot).pack(fill='x', expand=True, pady=5)
        
        self.fig2, ((self.ax2_0, self.ax2_1), (self.ax2_2, self.ax2_3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tab2)
        self.canvas2.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self.update_part2_plot()

    def update_part2_plot(self, event=None):
        x = np.arange(0, 20.1, 0.1)
        A = fuzz.gaussmf(x, self.p2_mean_a.get(), self.p2_sigma_a.get())
        B = fuzz.gaussmf(x, self.p2_mean_b.get(), self.p2_sigma_b.get())
        intersect_min = np.fmin(A, B); union_max = np.fmax(A, B); complement_A = 1 - A
        for ax in [self.ax2_0, self.ax2_1, self.ax2_2, self.ax2_3]: ax.clear()
            
        self.ax2_0.plot(x, A, 'b', linewidth=1.5, label='Множество A'); self.ax2_0.plot(x, B, 'g', linewidth=1.5, label='Множество B'); self.ax2_0.set_title("Исходные множества")
        self.ax2_1.fill_between(x, 0, intersect_min, facecolor='orange', alpha=0.7); self.ax2_1.plot(x, A, 'b:', linewidth=1); self.ax2_1.plot(x, B, 'g:', linewidth=1); self.ax2_1.set_title("Пересечение (min(A, B))")
        self.ax2_2.fill_between(x, 0, union_max, facecolor='cyan', alpha=0.7); self.ax2_2.plot(x, A, 'b:', linewidth=1); self.ax2_2.plot(x, B, 'g:', linewidth=1); self.ax2_2.set_title("Объединение (max(A, B))")
        self.ax2_3.plot(x, A, 'b:', linewidth=1, label='Множество A'); self.ax2_3.fill_between(x, 0, complement_A, facecolor='red', alpha=0.7, label='Дополнение A'); self.ax2_3.set_title("Дополнение (1 - A)")

        self.ax2_0.legend(); self.ax2_3.legend()
        for ax_ in self.fig2.get_axes(): ax_.grid(True); ax_.set_ylim(-0.05, 1.05)
        self.fig2.tight_layout()
        self.canvas2.draw()
        
    # ==============================================================================
    # ЧАСТЬ 3: НЕЧЕТКАЯ АППРОКСИМИРУЮЩАЯ СИСТЕМА
    # ==============================================================================
    def create_part3_ui(self):
        left_frame = ttk.Frame(self.tab3); left_frame.pack(side="left", fill="y", padx=10, pady=10)
        right_frame = ttk.Frame(self.tab3); right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        controls_frame = ttk.LabelFrame(left_frame, text="Тестирование системы"); controls_frame.pack(fill='x', pady=5, ipady=5)
        
        self.p3_input_x = tk.DoubleVar(value=-0.09)
        ttk.Label(controls_frame, text="Выберите входное значение 'x':").pack(pady=(10,0))
        ttk.Scale(controls_frame, from_=-1.0, to=1.0, orient='horizontal', variable=self.p3_input_x, command=self.p3_on_input_change).pack(fill='x', padx=10, pady=5)
        self.p3_output_label_var = tk.StringVar()
        ttk.Label(controls_frame, textvariable=self.p3_output_label_var, font=("Helvetica", 14), foreground="blue").pack(pady=20)
        
        self.fig3_sim, self.ax3_sim = plt.subplots(figsize=(6, 4))
        self.canvas3_sim = FigureCanvasTkAgg(self.fig3_sim, master=left_frame); self.canvas3_sim.get_tk_widget().pack(fill='both', expand=True, pady=(10,0))
        self.fig3_resp, self.ax3_resp = plt.subplots(figsize=(8, 6))
        self.canvas3_resp = FigureCanvasTkAgg(self.fig3_resp, master=right_frame); self.canvas3_resp.get_tk_widget().pack(fill='both', expand=True)
        
        self.x_points_v2 = np.array([-1, -0.6, 0.2, 0.4, 1])
        self.y_points_v2 = np.array([-1, -1.67, 5, 2.5, 1])
        self.p3_build_and_update_all()
        
    def p3_create_fis(self):
        x_input = ctrl.Antecedent(np.linspace(-1.1, 1.1, 301), 'x')
        y_output = ctrl.Consequent(np.linspace(-2, 5.5, 301), 'y')
        term_names = ['term1', 'term2', 'term3', 'term4', 'term5']
        centers = self.x_points_v2

        # --- ИСПРАВЛЕНИЕ ---
        # Возвращен ручной, но корректный способ создания ФП по заданным точкам
        for i, center_val in enumerate(centers):
            b = center_val
            a = centers[i - 1] if i > 0 else b
            c = centers[i + 1] if i < len(centers) - 1 else b
            x_input[term_names[i]] = fuzz.trimf(x_input.universe, [a, b, c])

        level_names = ['level_n1', 'level_n1_67', 'level_5', 'level_2_5', 'level_1']
        for i, value in enumerate(self.y_points_v2):
            y_output[level_names[i]] = fuzz.trimf(y_output.universe, [value, value, value])
        
        rules = [ctrl.Rule(x_input[term_names[i]], y_output[level_names[i]]) for i in range(len(term_names))]
        return ctrl.ControlSystem(rules), x_input

    def p3_build_and_update_all(self):
        self.p3_fis_control, self.p3_x_input = self.p3_create_fis()
        self.p3_update_response_curve_plot()
        self.p3_on_input_change() 

    def p3_on_input_change(self, event=None):
        fis_simulation = ctrl.ControlSystemSimulation(self.p3_fis_control)
        input_x = self.p3_input_x.get()
        fis_simulation.input['x'] = input_x
        
        try:
            fis_simulation.compute()
            output_y = fis_simulation.output.get('y')
            if output_y is not None:
                self.p3_output_label_var.set(f"Выход 'y' для x={input_x:.2f}:\n {output_y:.4f}")
            else:
                self.p3_output_label_var.set(f"Выход 'y' для x={input_x:.2f}:\n НЕ ВЫЧИСЛЕН")
        except Exception as e:
             self.p3_output_label_var.set(f"ОШИБКА ВЫЧИСЛЕНИЯ:\n{e}")

        self.p3_update_simulation_plot(input_x)

    def p3_update_simulation_plot(self, input_x):
        self.ax3_sim.clear()
        for term_name, term_obj in self.p3_x_input.terms.items():
            self.ax3_sim.plot(self.p3_x_input.universe, term_obj.mf, label=term_name)
            activation_level = fuzz.interp_membership(self.p3_x_input.universe, term_obj.mf, input_x)
            capped_mf = np.fmin(activation_level, term_obj.mf)
            self.ax3_sim.fill_between(self.p3_x_input.universe, 0, capped_mf, alpha=0.4)
        self.ax3_sim.axvline(x=input_x, color='k', linestyle='--', linewidth=2, label=f'Вход x={input_x:.2f}')
        self.ax3_sim.set_title(f"Активация ФП для x = {input_x:.2f}")
        self.ax3_sim.grid(True); self.ax3_sim.legend(fontsize='small'); self.ax3_sim.set_ylim(-0.05, 1.05)
        self.fig3_sim.tight_layout()
        self.canvas3_sim.draw()

    def p3_update_response_curve_plot(self):
        self.ax3_resp.clear()
        
        x_values = np.linspace(-1, 1, 201)
        y_values = np.interp(x_values, self.x_points_v2, self.y_points_v2)

        self.ax3_resp.plot(x_values, y_values, 'b', linewidth=2, label='Выход нечеткой системы')
        self.ax3_resp.plot(self.x_points_v2, self.y_points_v2, 'ro', markersize=8, label='Исходные точки (Вариант 2)')
        
        self.ax3_resp.set_title('Результат аппроксимации (Вариант 2)'); self.ax3_resp.set_xlabel('Вход: x'); self.ax3_resp.set_ylabel('Выход: y')
        self.ax3_resp.grid(True); self.ax3_resp.legend()
        self.fig3_resp.tight_layout()
        self.canvas3_resp.draw()

# ==============================================================================
# Точка входа
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = FuzzyLogicLabApp(root)
    root.mainloop()