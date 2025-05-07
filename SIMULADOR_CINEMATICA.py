import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import csv

class SimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulador de Movimiento: MRU, MRUA y Parabólico")
        self.geometry("1200x800")
        self.resizable(False, False)

        # fullscreen toggle state
        self.fullscreen = False
        self.bind("<F11>", lambda e: self.toggle_fullscreen())

        # tema ttk
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TLabel', font=('Arial', 11))
        style.configure('TButton', font=('Arial', 11), padding=5)
        style.configure('TRadiobutton', font=('Arial', 10))

        # simulación
        self.dt = 0.02
        self.running = False
        self.g = 9.81  # gravedad m/s²

        # variables de control
        self.mode = tk.StringVar(value="MRU")
        self.v0 = tk.DoubleVar(value=5.0)
        self.a = tk.DoubleVar(value=1.0)
        self.mu = tk.DoubleVar(value=0.0)
        self.angle = tk.DoubleVar(value=45.0)

        self._build_ui()

    def reset_data(self):
        # Sólo borrar rastro si el canvas ya existe
        if hasattr(self, 'canvas'):
            self.canvas.delete("trail")
            if self.mode.get() == 'Parabólico':
                self.canvas.delete('proj')
        # reiniciar variables del movimiento
        self.t = 0.0
        # condiciones iniciales para cada modo
        self.vx = self.v0.get() * np.cos(np.deg2rad(self.angle.get()))
        self.vy = self.v0.get() * np.sin(np.deg2rad(self.angle.get()))
        self.x = 0.0
        self.y = 0.0
        self.v = self.v0.get()
        # datos para graficar
        self.t_data = []
        self.x_data = []
        self.y_data = []
        self.v_data = []
        self.a_data = []

    def _build_ui(self):
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(ctrl, text="Tipo de movimiento:").pack(anchor=tk.W)
        for m in ("MRU", "MRUA", "Parabólico"):
            ttk.Radiobutton(ctrl, text=m, variable=self.mode, value=m).pack(anchor=tk.W)

        # velocidad inicial
        tk.Scale(ctrl, label="v₀ (m/s)", from_=0, to=20, resolution=0.1,
                 orient=tk.HORIZONTAL, variable=self.v0, tickinterval=5).pack(fill=tk.X, pady=5)
        # ángulo para parabólico
        tk.Scale(ctrl, label="Ángulo θ (°)", from_=0, to=90, resolution=1,
                 orient=tk.HORIZONTAL, variable=self.angle, tickinterval=15).pack(fill=tk.X, pady=5)
        # aceleración para MRUA
        tk.Scale(ctrl, label="a (m/s²)", from_=-5, to=5, resolution=0.1,
                 orient=tk.HORIZONTAL, variable=self.a, tickinterval=1).pack(fill=tk.X, pady=5)
        # fricción
        tk.Scale(ctrl, label="μ (rozamiento)", from_=0, to=1, resolution=0.01,
                 orient=tk.HORIZONTAL, variable=self.mu, tickinterval=0.2).pack(fill=tk.X, pady=5)

        # lecturas en tiempo real
        self.lbl_x = ttk.Label(ctrl, text="x = 0.00 m")
        self.lbl_y = ttk.Label(ctrl, text="y = 0.00 m")
        self.lbl_v = ttk.Label(ctrl, text="v = 0.00 m/s")
        self.lbl_a = ttk.Label(ctrl, text="a = 0.00 m/s²")
        self.lbl_x.pack(pady=(10,0))
        self.lbl_y.pack()
        self.lbl_v.pack()
        self.lbl_a.pack()

        # botones
        ttk.Button(ctrl, text="Guardar datos CSV", command=self.save_csv).pack(fill=tk.X, pady=(20,5))
        self.btn_full = ttk.Button(ctrl, text="Pantalla Completa (F11)", command=self.toggle_fullscreen)
        self.btn_full.pack(fill=tk.X, pady=5)
        self.btn_start = ttk.Button(ctrl, text="Iniciar", command=self.start)
        self.btn_start.pack(fill=tk.X, pady=5)
        self.btn_stop = ttk.Button(ctrl, text="Detener", command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X)

        # canvas animación con regla abajo
        self.canvas = tk.Canvas(self, width=600, height=300, bg="white")
        self.canvas.pack(side=tk.TOP, padx=10, pady=10)
        self._draw_ruler()
        # objeto móvil
        self.car = self.canvas.create_rectangle(0, 260, 40, 280, fill="blue", tags='proj')
        self.vel_arrow = self.canvas.create_line(0,0,0,0, arrow=tk.LAST, width=2, fill="red")
        self.acc_arrow = self.canvas.create_line(0,0,0,0, arrow=tk.LAST, width=2, fill="green")

        # pestañas de gráficos
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.axes, self.canvases = {}, {}

        for key, ylabel, title in [
            ("pos","Posición (m)","Posición vs tiempo"),
            ("vel","Velocidad (m/s)","Velocidad vs tiempo"),
            ("acc","Aceleración (m/s²)","Aceleración vs tiempo"),
            ("ek","Energía cinética (J)","Energía cinética vs tiempo")
        ]:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=title)
            fig = Figure(figsize=(6,3), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_title(title)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            canvas_fig = FigureCanvasTkAgg(fig, master=frame)
            canvas_fig.get_tk_widget().pack(expand=True, fill=tk.BOTH)
            self.axes[key] = ax
            self.canvases[key] = canvas_fig

    def _draw_ruler(self):
        w = int(self.canvas['width'])
        y0 = 270
        self.canvas.create_line(0, y0, w, y0, fill="black")
        px_per_m = 30
        for i in range(0, w//px_per_m+1):
            x = i*px_per_m
            self.canvas.create_line(x, y0, x, y0+10)
            self.canvas.create_text(x, y0+20, text=f"{i} m")

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.attributes("-fullscreen", self.fullscreen)

    def start(self):
        if self.running: return
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.reset_data()
        self._animate()

    def stop(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self._update_graphs()

    def _animate(self):
        if not self.running: return
        mode = self.mode.get()
        a = self.a.get()
        mu = self.mu.get()

        if mode == 'MRU':
            a_net = -mu*self.v
            self.x += self.v*self.dt
            self.v += a_net*self.dt
            self.y = 0
        elif mode == 'MRUA':
            a_net = a - mu*self.v
            self.x += self.v*self.dt + 0.5*a_net*self.dt**2
            self.v += a_net*self.dt
            self.y = 0
        else:  # Parabólico
            a_net_x = -mu*self.vx
            a_net_y = -self.g - mu*self.vy
            self.x += self.vx*self.dt
            self.y += self.vy*self.dt + 0.5*(-self.g)*self.dt**2
            self.vx += a_net_x*self.dt
            self.vy += a_net_y*self.dt
            a_net = np.hypot(a_net_x, a_net_y)

        self.t += self.dt
        self.t_data.append(self.t)
        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.v_data.append(self.v if mode!='Parabólico' else np.hypot(self.vx, self.vy))
        self.a_data.append(a_net)

        # rastro
        px = self.x*30; py = 260 - self.y*30
        self.canvas.create_oval(px+18, py+8, px+22, py+12, fill='blue', outline='', tags="trail")
        # dibuja objeto
        if mode=='Parabólico':
            self.canvas.coords('proj', px, py, px+20, py+20)
        else:
            self.canvas.coords(self.car, px, 260, px+40, 280)
        # flechas
        cx, cy = px+20, (py+10 if mode=='Parabólico' else 260)
        self.canvas.coords(self.vel_arrow, cx, cy, cx + (self.vx if mode=='Parabólico' else self.v)*10, cy)
        self.canvas.coords(self.acc_arrow, cx, cy, cx + a_net*50, cy)

        # lecturas
        self.lbl_x.config(text=f"x = {self.x:.2f} m")
        self.lbl_y.config(text=f"y = {self.y:.2f} m")
        self.lbl_v.config(text=f"v = {(self.v if mode!='Parabólico' else np.hypot(self.vx,self.vy)):.2f} m/s")
        self.lbl_a.config(text=f"a = {a_net:.2f} m/s²")

        # condiciones de parada
        if px>600 or px<0 or py>300 or py<0:
            self.stop(); return
        self.after(int(self.dt*1000), self._animate)

    def _update_graphs(self):
        # posición: x e y
        ax = self.axes['pos']; ax.clear()
        ax.plot(self.t_data, self.x_data, label='x')
        ax.plot(self.t_data, self.y_data, label='y')
        ax.legend(); ax.grid(True)
        self.canvases['pos'].draw()
        # otros
        for key in ('vel','acc'):
            y = {'vel':self.v_data,'acc':self.a_data}[key]
            ax = self.axes[key]; ax.clear(); ax.plot(self.t_data, y);
            ax.grid(True); self.canvases[key].draw()
        # energía cinética
        ek = [0.5*(v**2) for v in self.v_data]
        ax = self.axes['ek']; ax.clear(); ax.plot(self.t_data, ek); ax.grid(True); self.canvases['ek'].draw()

    def save_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path: return
        with open(path,'w',newline='') as f:
            w=csv.writer(f); w.writerow(['t','x','y','v','a'])
            w.writerows(zip(self.t_data,self.x_data,self.y_data,self.v_data,self.a_data))

if __name__ == "__main__":
    SimulatorApp().mainloop()
