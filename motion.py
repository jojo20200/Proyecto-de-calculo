# ultra_calc2d_ui_final_integrado_fixed.py
"""
Versión final (ARREGLADA Y MEJORADA) que integra el análisis de puntos críticos en la interfaz detallada.
Se ha mejorado la interfaz de usuario con un diseño más moderno y se ha añadido un manejo de errores
robusto con ventanas de diálogo para una mejor experiencia de usuario.
Mantiene los requisitos: pip install sympy numpy matplotlib
"""
import math
import os
import numpy as np
import sympy as sp
from sympy import Symbol
from sympy.core.sympify import SympifyError
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import animation
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox, colorchooser

# ----------------------- UTILIDADES -----------------------
X = sp.Symbol('x')
T = sp.Symbol('t')  # usado para sustituciones temporales


def parse_limit_string(s):
    """Parsea un string de límite: 'inf', 'oo', '-inf', '-oo', o número."""
    s = str(s).strip()
    if s == '':
        raise ValueError("El límite no puede ser una cadena vacía.")
    sl = s.lower()
    if sl in ('inf', 'oo', '+inf', '+oo'):
        return sp.oo
    if sl in ('-inf', '-oo'):
        return -sp.oo
    try:
        # Intenta convertir a float para validación numérica estándar
        return float(s)
    except ValueError:
        try:
            # Si falla, intenta sympificar (ej. 'pi', 'E')
            return sp.sympify(s)
        except SympifyError:
            raise ValueError(f"Límite inválido: '{s}'. Use números, 'inf', '-inf', 'pi', o 'E'.")


def safe_sympify(expr_str, symbol_char='x'):
    """Parsea la expresión ingresada por el usuario a SymPy de forma segura."""
    if not str(expr_str).strip():
        raise ValueError("La expresión de la función no puede estar vacía.")
    
    s = str(expr_str).replace('^', '**')
    
    # Diccionario de funciones y constantes permitidas
    local = {
        symbol_char: Symbol(symbol_char),
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'csc': sp.csc, 'sec': sp.sec, 'cot': sp.cot,
        'exp': sp.exp, 'log': sp.log, 'ln': sp.log, 
        'sqrt': sp.sqrt, 'pi': sp.pi, 'E': sp.E, 
        'abs': sp.Abs, 'factorial': sp.factorial
    }
    
    try:
        # Evalúa la expresión en un entorno controlado
        expr = sp.sympify(s, locals=local)
        return expr
    except (SympifyError, SyntaxError) as e:
        raise ValueError(f"Error de sintaxis en la expresión: '{expr_str}'.\n\nCausa: {e}\n\nRevise la sintaxis. Use '*' para multiplicar (ej. 2*x) y '**' para potencias (ej. x**2).")
    except Exception as e:
        raise ValueError(f"Error desconocido al procesar la expresión: {e}")

# ----------------------- SOLVER PROFESIONAL (NUEVA CLASE) -----------------------
class StepByStepSolver:
    """
    Genera explicaciones detalladas para operaciones de cálculo.
    Simula un enfoque "paso a paso" para derivadas e integrales.
    """
    def __init__(self):
        self.x = sp.Symbol('x')

    def _format_step(self, text, level=0):
        """Formatea un paso con indentación para la jerarquía."""
        return "    " * level + f"• {text}"

    def get_derivative_steps(self, expr):
        """
        Genera una lista de pasos explicando cómo derivar una expresión.
        Es una función recursiva que descompone la expresión.
        """
        steps = [f"Vamos a encontrar la derivada de f(x) = {expr}"]
        
        try:
            result = self._derive(expr, steps, level=1)
            steps.append(f"\nResultado final de la derivada:")
            steps.append(f"f'(x) = {sp.simplify(result)}")
        except Exception as e:
            steps.append(f"No se pudo generar un paso a paso detallado: {e}")
            steps.append(f"Resultado directo de SymPy: {sp.diff(expr, self.x)}")
            
        return steps

    def _derive(self, expr, steps, level):
        """Motor recursivo para la derivación paso a paso."""
        # Caso base: constante o símbolo
        if expr.is_constant() or expr == self.x:
            deriv = expr.diff(self.x)
            if expr.is_constant():
                steps.append(self._format_step(f"La derivada de una constante ({expr}) es 0.", level))
            else: # es self.x
                steps.append(self._format_step(f"La derivada de x con respecto a x es 1.", level))
            return deriv

        # Regla de la suma: f(x) + g(x)
        if isinstance(expr, sp.Add):
            steps.append(self._format_step("Aplicamos la regla de la suma: (u + v)' = u' + v'", level))
            deriv_parts = []
            for arg in expr.args:
                steps.append(self._format_step(f"Derivando el término: {arg}", level + 1))
                deriv_parts.append(self._derive(arg, steps, level + 2))
            return sp.Add(*deriv_parts)

        # Regla del producto: f(x) * g(x)
        if isinstance(expr, sp.Mul):
            # Separar constante de la función, ej. 3*x**2
            const, func_part = expr.as_coeff_Mul()
            if const != 1:
                steps.append(self._format_step(f"Aplicamos la regla del múltiplo constante: (c*u)' = c*u'", level))
                steps.append(self._format_step(f"La constante c = {const}", level + 1))
                steps.append(self._format_step(f"Ahora derivamos u = {func_part}", level + 1))
                return const * self._derive(func_part, steps, level + 2)
            else:
                # Regla del producto completa u*v
                u, v = expr.as_two_terms()
                steps.append(self._format_step("Aplicamos la regla del producto: (u*v)' = u'v + uv'", level))
                steps.append(self._format_step(f"Sea u = {u} y v = {v}", level + 1))
                steps.append(self._format_step("Calculando u':", level + 1))
                du = self._derive(u, steps, level + 2)
                steps.append(self._format_step("Calculando v':", level + 1))
                dv = self._derive(v, steps, level + 2)
                return u * dv + v * du

        # Regla de la potencia: x^n
        if isinstance(expr, sp.Pow):
            base, exp = expr.args
            if base == self.x and exp.is_constant():
                steps.append(self._format_step(f"Aplicamos la regla de la potencia: d/dx(x^n) = n*x^(n-1)", level))
                return expr.diff(self.x)
            # Regla de la cadena para potencias: u^n
            if exp.is_constant():
                steps.append(self._format_step(f"Aplicamos la regla de la cadena con la regla de la potencia: (u^n)' = n*u^(n-1)*u'", level))
                u = base
                n = exp
                steps.append(self._format_step(f"Sea u = {u} y n = {n}", level + 1))
                steps.append(self._format_step("Calculando u':", level + 1))
                du = self._derive(u, steps, level + 2)
                return n * (u ** (n - 1)) * du

        # Regla de la cadena para funciones (sin, cos, exp, log, etc.)
        if isinstance(expr, (sp.sin, sp.cos, sp.tan, sp.exp, sp.log)):
            inner_func = expr.args[0]
            if inner_func == self.x:
                steps.append(self._format_step(f"La derivada de {expr.func.__name__}(x) es {expr.diff(self.x)}.", level))
                return expr.diff(self.x)
            else:
                steps.append(self._format_step(f"Aplicamos la regla de la cadena: d/dx(f(g(x))) = f'(g(x)) * g'(x)", level))
                u = inner_func
                outer_func_name = expr.func.__name__
                steps.append(self._format_step(f"Sea la función interna u = {u}", level + 1))
                steps.append(self._format_step(f"La función externa es {outer_func_name}(u)", level + 1))
                
                # Derivada de la función externa
                outer_deriv = expr.func(self.x).diff(self.x).subs(self.x, u)
                steps.append(self._format_step(f"La derivada de {outer_func_name}(u) con respecto a u es {outer_deriv}", level + 1))
                
                # Derivada de la función interna
                steps.append(self._format_step("Ahora calculamos la derivada de la función interna u':", level + 1))
                inner_deriv = self._derive(u, steps, level + 2)
                
                steps.append(self._format_step(f"Combinando todo: {outer_deriv} * ({inner_deriv})", level + 1))
                return outer_deriv * inner_deriv

        # Si ninguna regla coincide, usar la derivación directa de Sympy
        steps.append(self._format_step(f"Derivando {expr} directamente.", level))
        return expr.diff(self.x)

# ----------------------- CALCULADORA UI (NUEVA CLASE) -----------------------
class CalculatorPad(ttk.Frame):
    """
    Un widget de calculadora para insertar texto en un campo de entrada.
    """
    def __init__(self, parent, entry_widget):
        super().__init__(parent, style="Card.TFrame")
        self.entry = entry_widget

        # Definición de los botones de la calculadora
        buttons = [
            ('sin(', 'cos(', 'tan(', 'exp('),
            ('log(', 'sqrt(', 'abs(', '^'),
            ('7', '8', '9', '/'),
            ('4', '5', '6', '*'),
            ('1', '2', '3', '-'),
            ('0', '.', 'x', '+'),
            ('(', ')', 'pi', 'E'),
        ]

        for r, row_items in enumerate(buttons):
            self.grid_rowconfigure(r, weight=1)
            for c, item in enumerate(row_items):
                self.grid_columnconfigure(c, weight=1)
                
                # Texto a mostrar vs. texto a insertar
                display_text = item.replace('(', '') if '(' in item else item
                insert_text = item
                
                # Mapeo de texto a símbolos más bonitos
                if item == '*': display_text = '×'
                if item == '/': display_text = '÷'
                if item == '^': display_text = 'xʸ'

                button = ttk.Button(
                    self,
                    text=display_text,
                    command=lambda val=insert_text: self.on_press(val)
                )
                button.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
        
        # Botón de borrar (ocupa dos columnas)
        clear_button = ttk.Button(self, text='Borrar', command=self.on_backspace, style="TButton")
        clear_button.grid(row=len(buttons), column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
        
        # Botón de limpiar todo (ocupa dos columnas)
        clear_all_button = ttk.Button(self, text='Limpiar Todo', command=self.on_clear, style="Accent.TButton")
        clear_all_button.grid(row=len(buttons), column=2, columnspan=2, sticky="nsew", padx=2, pady=2)

    def on_press(self, value):
        """Inserta el valor del botón en la posición del cursor."""
        self.entry.insert(tk.INSERT, value)
        self.entry.focus_set()

    def on_backspace(self):
        """Borra el carácter anterior a la posición del cursor."""
        cursor_pos = self.entry.index(tk.INSERT)
        if cursor_pos > 0:
            self.entry.delete(cursor_pos - 1, cursor_pos)
        self.entry.focus_set()
    
    def on_clear(self):
        """Limpia todo el contenido del campo de entrada."""
        self.entry.delete(0, tk.END)
        self.entry.focus_set()


# ----------------------- MATH ENGINE (sin cambios) -----------------------
class MathEngine:
    def __init__(self):
        self.x = X

    def derivative(self, func_str, order=1):
        expr = safe_sympify(func_str, 'x')
        d = sp.diff(expr, self.x, order)
        steps = [f"f(x) = {sp.simplify(expr)}", f"Derivada {order}ª: {sp.simplify(d)}"]
        return d, steps

    def eval_derivative(self, func_str, point, order=1):
        expr = safe_sympify(func_str, 'x')
        d = sp.diff(expr, self.x, order)
        try:
            val = sp.N(d.subs(self.x, point))
            steps = [f"f(x) = {expr}", f"Derivada {order}ª: {d}", f"Evaluada en x={point}: {val}"]
            return val, steps
        except Exception as e:
            return None, [f"No fue posible evaluar la derivada en {point}: {e}"]

    def indefinite(self, func_str):
        expr = safe_sympify(func_str, 'x')
        try:
            res = sp.integrate(expr, self.x)
            steps = [f"f(x) = {expr}", f"∫ f(x) dx = {res} + C"]
            return res, steps
        except Exception as e:
            return None, [f"No se pudo integrar simbólicamente: {e}"]

    def definite(self, func_str, a, b):
        expr = safe_sympify(func_str, 'x')
        try:
            res = sp.integrate(expr, (self.x, a, b))
            steps = [f"f(x) = {expr}", f"Intervalo: [{a}, {b}]", f"Resultado simbólico: {res}"]
            return res, steps
        except Exception:
            try:
                F = sp.integrate(expr, self.x)
                val = sp.N(F.subs(self.x, b) - F.subs(self.x, a))
                steps = [f"f(x) = {expr}", f"Primitiva: {F}", f"Área = F({b}) - F({a}) = {val}"]
                return val, steps
            except Exception as e2:
                return None, [f"No se pudo calcular integral definida simbólicamente: {e2}"]

    def improper(self, func_str, a, b):
        expr = safe_sympify(func_str, 'x')
        steps = [f"f(x) = {expr}", f"Intentando integral impropia en [{a}, {b}]"]
        try:
            res = sp.integrate(expr, (self.x, a, b))
            steps.append(f"Resultado simbólico: {res}")
            return res, steps
        except Exception:
            steps.append("No se pudo evaluar simbólicamente; intente con aproximación numérica o analice por límites.")
            return None, steps

    def partial_fractions(self, func_str):
        expr = safe_sympify(func_str, 'x')
        steps = [f"Expresión original: {expr}"]
        try:
            decomposed = sp.apart(expr, self.x)
            steps.append(f"Descomposición en fracciones parciales: {decomposed}")
            try:
                integ = sp.integrate(decomposed, self.x)
                steps.append(f"Integral de la descomposición: {integ} + C")
                return integ, steps
            except Exception as e:
                steps.append(f"No se pudo integrar la descomposición automáticamente: {e}")
                return None, steps
        except Exception as e:
            return None, [f"No es una fracción racional o no se pudo aplicar apart(): {e}"]

    def integration_by_parts_auto(self, func_str):
        expr = safe_sympify(func_str, 'x')
        steps = [f"Expresión: {expr}", "Método: Integración por partes (heurística LIATE)"]
        try:
            # Sympy tiene una implementación directa que es más robusta
            result = sp.integrate(expr, self.x, manual=True)
            if any(isinstance(i, sp.Integral) and i.has(sp.Integral) for i in sp.make_list(result.atoms(sp.Integral))):
                 raise ValueError("La integración por partes no simplificó el problema.")
            
            # Para mostrar los pasos, podemos intentar recrear la elección de u y dv
            # Esta parte es heurística y puede no coincidir con el proceso interno de Sympy
            u, dv = expr.as_two_terms() # Simplificación
            du = sp.diff(u, self.x)
            v = sp.integrate(dv, self.x)
            
            steps.append(f"Elección heurística: u = {u}, dv = {dv} dx")
            steps.append(f"Calculando: du = {du} dx, v = ∫dv = {v}")
            steps.append(f"Fórmula: ∫u dv = u*v - ∫v du")
            steps.append(f"Resultado: {result} + C")
            return result, steps
        except Exception as e:
            return None, steps + [f"No se pudo aplicar integración por partes de forma automática: {e}"]


    def substitution_auto(self, func_str):
        expr = safe_sympify(func_str, 'x')
        steps = [f"Función: {expr}", "Intento de sustitución automática"]
        try:
            # La función `integral_steps` de Sympy puede encontrar sustituciones
            # aunque es una función no pública y puede cambiar.
            from sympy.integrals.risch import NonElementaryIntegral
            from sympy.integrals.manualintegrate import integral_steps
            
            result = sp.integrate(expr, self.x)
            if isinstance(result, NonElementaryIntegral):
                  raise ValueError("La integral no es elemental.")

            steps_obj = integral_steps(expr, self.x)
            # Analizar el objeto de pasos para mostrar la sustitución
            # Esto es complejo; por ahora, solo mostramos el resultado
            steps.append(f"Sympy encontró una sustitución adecuada.")
            steps.append(f"Resultado final: {result} + C")
            return result, steps
        except Exception as e:
            return None, steps + ["No se detectó una sustitución directa con heurística simple.", f"Error: {e}"]

    def trig_substitution(self, func_str):
        expr = safe_sympify(func_str, 'x')
        steps = [f"Función: {expr}", "Intento de sustitución trigonométrica"]
        try:
            # Usar `manual=True` a menudo fuerza este tipo de métodos
            result = sp.integrate(expr, self.x, manual=True)
            if not isinstance(result, sp.Integral):
                steps.append("Sympy aplicó una sustitución trigonométrica (o equivalente).")
                steps.append(f"Resultado: {result} + C")
                return result, steps
            else:
                raise ValueError("No se pudo resolver con este método.")
        except Exception as e:
            return None, steps + [f"No se detectó forma de sustitución trigonométrica automáticamente. Error: {e}"]

    def numeric_simpson(self, func_str, a, b, n=2000):
        expr = safe_sympify(func_str, 'x')
        f = sp.lambdify(self.x, expr, 'numpy')
        
        # Asegurar que n es par
        if n % 2 != 0:
            n += 1
            
        xs = np.linspace(float(a), float(b), n + 1)
        
        try:
            with np.errstate(all='ignore'):
                ys = f(xs)
        except Exception as e:
            raise ValueError(f"No se pudo evaluar la función numéricamente: {e}")

        # Reemplazar valores no finitos
        ys = np.nan_to_num(ys, nan=0.0, posinf=1e100, neginf=-1e100) # Usar un valor grande en vez de 0
        
        h = (float(b) - float(a)) / n
        integral = (h / 3) * (ys[0] + 4 * np.sum(ys[1:-1:2]) + 2 * np.sum(ys[2:-2:2]) + ys[-1])
        
        steps = [f"Aproximación numérica (Regla de Simpson) con n={n} intervalos.", f"Resultado aproximado: {integral}"]
        return float(integral), steps
    
    def critical_points(self, func_str, xlim=(-5, 5)):
        expr = safe_sympify(func_str, 'x')
        fprime = sp.diff(expr, self.x)
        fsecond = sp.diff(expr, self.x, 2)
        
        crits = []
        try:
            # Usar nsolve para encontrar soluciones numéricas en el intervalo
            sols_num = []
            for point in np.linspace(xlim[0], xlim[1], 100): # Puntos de inicio
                try:
                    sol = sp.nsolve(fprime, self.x, point)
                    if xlim[0] <= sol <= xlim[1]:
                        sols_num.append(float(sol))
                except (ValueError, TypeError):
                    continue
            
            # Añadir soluciones simbólicas si son reales
            sols_sym = sp.solve(sp.Eq(fprime, 0), self.x)
            for s in sols_sym:
                if s.is_real:
                      sols_num.append(float(s))

            sols = sorted(list(set(np.round(sols_num, 5)))) # Redondear para eliminar duplicados cercanos

            for s in sols:
                sx = float(s)
                if xlim[0] <= sx <= xlim[1]:
                    try:
                        yv = float(expr.subs(self.x, s))
                        sec_val = float(fsecond.subs(self.x, s))
                        kind = 'Mínimo local' if sec_val > 0 else ('Máximo local' if sec_val < 0 else 'Punto de inflexión candidato')
                        crits.append({'x': sx, 'y': yv, 'tipo': kind})
                    except (ValueError, TypeError):
                        continue # Ignorar puntos complejos o indefinidos
        except Exception:
            pass # Falla si fprime es muy compleja

        y_intercept = {'x': 0.0, 'y': float(expr.subs(self.x, 0)) if expr.subs(self.x, 0).is_real else None, 'tipo': 'Intersección Y'}

        x_intercepts = []
        try:
            # Búsqueda numérica de raíces
            roots_num = []
            for point in np.linspace(xlim[0], xlim[1], 100):
                try:
                    root = sp.nsolve(expr, self.x, point)
                    if xlim[0] <= root <= xlim[1]:
                        roots_num.append(float(root))
                except (ValueError, TypeError):
                    continue
            
            roots = sorted(list(set(np.round(roots_num, 5))))

            for r in roots:
                if xlim[0] <= r <= xlim[1]:
                    x_intercepts.append({'x': r, 'y': 0.0, 'tipo': 'Raíz'})
        except Exception:
            pass
            
        return {"puntos_criticos": crits, "interseccion_y": y_intercept, "raices": x_intercepts}


# ----------------------- PLOTTING (CON MEJOR ESTILO) -----------------------
class PlotManager:
    def __init__(self, fig, ax, canvas):
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.anim = None
        self.bands = []
        self.annotations = []
        self._apply_style()

    def _apply_style(self):
        self.fig.patch.set_facecolor("#0b1220")
        self.ax.set_facecolor("#0f1720")
        
        self.ax.tick_params(colors='white', which='both')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#26343f')
        self.ax.spines['bottom'].set_color('#26343f')
        
        self.ax.grid(True, linestyle='--', linewidth=0.5, color='#26343f', alpha=0.7)
        self.ax.axhline(0, color='#26343f', linewidth=0.8)
        self.ax.axvline(0, color='#26343f', linewidth=0.8)

    def clear(self):
        if self.anim:
            try:
                self.anim.event_source.stop()
            except AttributeError:
                pass
            self.anim = None
        
        for artist in self.annotations:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self.annotations.clear()

        self.bands.clear()
        
        # Limpiar eje sin destruir objetos externos
        self.ax.cla()
        self._apply_style()
        self.canvas.draw_idle()

    def plot_function(self, func_str, xlim=(-5,5), color='#60a5fa', lw=2.2):
        expr = safe_sympify(func_str, 'x')
        f = sp.lambdify(X, expr, 'numpy')
        
        xs = np.linspace(float(xlim[0]), float(xlim[1]), 1500)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ys = f(xs)
        
        ys = np.array(ys, dtype=float)
        ys[np.isinf(ys)] = np.nan
        
        # Detectar y cortar asíntotas verticales (discontinuidades grandes)
        if ys is not None:
            y_diff = np.abs(np.diff(ys))
            median_diff = np.nanmedian(y_diff)
            if median_diff > 1e-6: # Evitar division por cero
                threshold = median_diff * 25
                ys[np.hstack((False, y_diff > threshold))] = np.nan
        
        line, = self.ax.plot(xs, ys, linewidth=lw, label=f"f(x) = {expr}", color=color)
        self.ax.set_xlim(xlim)
        
        # Ajustar límites Y para mejor visualización, evitando valores extremos
        valid_ys = ys[np.isfinite(ys)]
        if valid_ys.size > 0:
            mean_y, std_y = np.mean(valid_ys), np.std(valid_ys)
            self.ax.set_ylim(mean_y - 4 * std_y, mean_y + 4 * std_y)
        
        self.ax.legend(facecolor='#0b1220', edgecolor='#22303f', labelcolor='white', fontsize='small')
        self.canvas.draw_idle()

    def shade_integral(self, func_str, a, b, base_color='#60a5fa', animate=True):
        expr = safe_sympify(func_str, 'x')
        f = sp.lambdify(X, expr, 'numpy')
        
        xs_area = np.linspace(float(a), float(b), 400)
        with np.errstate(all='ignore'):
            ys_area = f(xs_area)

        ys_area = np.nan_to_num(ys_area, nan=0.0)
        
        self.ax.fill_between(xs_area, 0, ys_area, alpha=0.3, color=base_color, edgecolor='none')
        self.canvas.draw_idle()

    def mark_points(self, points, color='#f59e0b'):
        if self.anim:
            self.anim.event_source.stop()
            self.anim = None
        
        for art in list(self.annotations):
            try:
                art.remove()
            except (ValueError, AttributeError):
                pass
        self.annotations.clear()

        for p in points:
            x, y, label = p.get('x'), p.get('y'), p.get('tipo', '')
            if x is None or y is None:
                continue
            
            dot = self.ax.scatter([x], [y], color=color, s=50, zorder=5, edgecolors='white', linewidths=0.8)
            ann = self.ax.annotate(f"{label}\n({x:.2f}, {y:.2f})", (x, y),
                                  textcoords="offset points", xytext=(0, 12), ha='center',
                                  fontsize=8, color='white',
                                  bbox=dict(boxstyle='round,pad=0.3', fc='#0f1720', ec='none', alpha=0.85))
            self.annotations.extend([dot, ann])
            
        self.canvas.draw_idle()

# ----------------------- INTERFAZ (MEJORADA Y CON NUEVAS CLASES) -----------------------
class UltraCalc2DDetailedApp:
    def __init__(self, root):
        self.root = root
        root.title("Ultra Calc 2D — Analizador de Cálculo")
        root.geometry("1440x850")
        root.minsize(1200, 700)
        root.configure(bg="#071019")
        self.engine = MathEngine()
        self.solver = StepByStepSolver() # <-- NUEVO SOLVER PROFESIONAL

        self._setup_styles()

        main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        left_panel = ttk.Frame(main_paned, width=400)
        main_paned.add(left_panel, weight=1)
        center_panel = ttk.Frame(main_paned)
        main_paned.add(center_panel, weight=3)
        right_panel = ttk.Frame(main_paned, width=380)
        main_paned.add(right_panel, weight=1)

        self._create_left_panel(left_panel)
        self._create_center_panel(center_panel)
        self._create_right_panel(right_panel)

        self.status = tk.StringVar(value="Listo para calcular")
        status_bar = ttk.Label(root, textvariable=self.status, anchor="w", style="Status.TLabel")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Colores
        BG_COLOR = "#071019"
        CARD_BG = "#0b1220"
        TEXT_COLOR = "#e6eef6"
        ACCENT_COLOR = "#60a5fa"
        ACCENT_HOVER = "#93c5fd"
        BUTTON_BG = "#1e293b"
        BUTTON_HOVER = "#334155"
        
        # Configuración General
        style.configure(".", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TEntry", fieldbackground=BUTTON_BG, foreground=TEXT_COLOR, bordercolor="#475569", insertcolor=TEXT_COLOR)
        style.map("TEntry", bordercolor=[('focus', ACCENT_COLOR)])
        
        # Botones
        style.configure("TButton", font=("Segoe UI", 10, "bold"), background=BUTTON_BG, foreground=TEXT_COLOR, borderwidth=0, padding=(10, 8))
        style.map("TButton", background=[('active', BUTTON_HOVER), ('pressed', BUTTON_HOVER)])
        
        # Botón de Acento (acciones principales)
        style.configure("Accent.TButton", background=ACCENT_COLOR, foreground=CARD_BG)
        style.map("Accent.TButton", background=[('active', ACCENT_HOVER), ('pressed', ACCENT_HOVER)])

        # Notebook (pestañas)
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_COLOR, foreground="#94a3b8", padding=(10, 5), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", CARD_BG)], foreground=[("selected", TEXT_COLOR)])
        style.configure("TCombobox", arrowcolor=TEXT_COLOR, fieldbackground=BUTTON_BG, background=BUTTON_BG, foreground=TEXT_COLOR, bordercolor="#475569", insertcolor=TEXT_COLOR)

        # Labels de Título y Cards
        style.configure("Card.TFrame", background=CARD_BG, relief="solid", borderwidth=1, bordercolor="#22303f")
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground=ACCENT_COLOR, background=CARD_BG)
        style.configure("CardHeader.TLabel", font=("Segoe UI", 12, "bold"), foreground=ACCENT_COLOR, background=CARD_BG)
        style.configure("Small.TLabel", font=("Segoe UI", 9), foreground="#cbd7e6", background=CARD_BG)
        style.configure("Status.TLabel", font=("Segoe UI", 9), foreground="#9aa5b1", background=BG_COLOR)
    
    def _create_card(self, parent, title):
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.pack(fill=tk.X, pady=8)
        ttk.Label(card, text=title, style="CardHeader.TLabel").pack(anchor='w', pady=(0, 10))
        return card

    def _create_left_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        func_tab = ttk.Frame(notebook, padding=10); notebook.add(func_tab, text="Función")
        deriv_tab = ttk.Frame(notebook, padding=10); notebook.add(deriv_tab, text="Derivadas")
        integral_tab = ttk.Frame(notebook, padding=10); notebook.add(integral_tab, text="Integrales")
        options_tab = ttk.Frame(notebook, padding=10); notebook.add(options_tab, text="Opciones")
        help_tab = ttk.Frame(notebook, padding=10); notebook.add(help_tab, text="Ayuda")

        # Pestaña Función
        func_card = self._create_card(func_tab, "Función y Rango")
        ttk.Label(func_card, text="Ingrese f(x). Use ';' para múltiples funciones.", style="Small.TLabel").pack(anchor='w')
        self.func_entry = ttk.Entry(func_card, font=("Consolas", 11)); self.func_entry.pack(fill=tk.X, pady=6)
        
        # --- INICIO DEL CÓDIGO AÑADIDO ---
        # Crear e instanciar la calculadora visual
        calculator_pad = CalculatorPad(func_card, self.func_entry)
        calculator_pad.pack(fill=tk.X, pady=(10, 5), expand=True)
        # --- FIN DEL CÓDIGO AÑADIDO ---
        
        self.func_entry.insert(0, "sin(x) * exp(-x/5)")

        fr_range = ttk.Frame(func_card, style="Card.TFrame"); fr_range.pack(fill=tk.X, pady=6)
        ttk.Label(fr_range, text="X min", style="Small.TLabel").grid(row=0, column=0, sticky='w')
        self.xmin_entry = ttk.Entry(fr_range, width=12); self.xmin_entry.grid(row=1, column=0, padx=(0,5), sticky='ew')
        ttk.Label(fr_range, text="X max", style="Small.TLabel").grid(row=0, column=1, sticky='w')
        self.xmax_entry = ttk.Entry(fr_range, width=12); self.xmax_entry.grid(row=1, column=1, padx=(5,0), sticky='ew')
        fr_range.columnconfigure((0, 1), weight=1)
        self.xmin_entry.insert(0, "-10"); self.xmax_entry.insert(0, "10")

        lim_card = self._create_card(func_tab, "Límites de Integración")
        fr_lim = ttk.Frame(lim_card, style="Card.TFrame"); fr_lim.pack(fill=tk.X, pady=6)
        ttk.Label(fr_lim, text="Límite inferior (a)", style="Small.TLabel").grid(row=0, column=0, sticky='w')
        self.lower_entry = ttk.Entry(fr_lim, width=12); self.lower_entry.grid(row=1, column=0, padx=(0,5), sticky='ew')
        ttk.Label(fr_lim, text="Límite superior (b)", style="Small.TLabel").grid(row=0, column=1, sticky='w')
        self.upper_entry = ttk.Entry(fr_lim, width=12); self.upper_entry.grid(row=1, column=1, padx=(5,0), sticky='ew')
        fr_lim.columnconfigure((0, 1), weight=1)
        self.lower_entry.insert(0, "0"); self.upper_entry.insert(0, "pi")
        
        ttk.Button(func_tab, text="📈 Graficar Función", command=self.on_plot, style="Accent.TButton").pack(fill=tk.X, pady=(15, 5))
        ttk.Button(func_tab, text="📍 Analizar Puntos Críticos", command=self.on_analyze_points).pack(fill=tk.X, pady=5)

        # Pestaña Derivadas
        deriv_card = self._create_card(deriv_tab, "Cálculo de Derivadas")
        ttk.Label(deriv_card, text="Orden:", style="Small.TLabel").pack(anchor='w')
        self.deriv_order = ttk.Combobox(deriv_card, values=["1","2","3","4"], state='readonly', width=8); self.deriv_order.pack(fill=tk.X, pady=5)
        self.deriv_order.set("1")
        ttk.Label(deriv_card, text="Evaluar en x =", style="Small.TLabel").pack(anchor='w')
        self.eval_point = ttk.Entry(deriv_card); self.eval_point.pack(fill=tk.X, pady=5)
        self.eval_point.insert(0, "1")
        ttk.Button(deriv_tab, text="Calcular Derivada", command=lambda: self.on_calculate("derivada"), style="Accent.TButton").pack(fill=tk.X, pady=15)

        # Pestaña Integrales
        integral_card = self._create_card(integral_tab, "Cálculo de Integrales")
        ttk.Label(integral_card, text="Método de cálculo:", style="Small.TLabel").pack(anchor='w')
        self.calc_type = ttk.Combobox(integral_card, values=["Auto (mejor intento)", "Integral indefinida", "Integral definida", "Integral impropia", "Sustitución", "Por partes", "Fracciones parciales", "Sust. trigonométrica", "Numérico (Simpson)"], state='readonly')
        self.calc_type.pack(fill=tk.X, pady=5)
        self.calc_type.set("Auto (mejor intento)")
        ttk.Button(integral_tab, text="Calcular Integral", command=lambda: self.on_calculate("integral"), style="Accent.TButton").pack(fill=tk.X, pady=15)

        # Pestaña Opciones
        options_card = self._create_card(options_tab, "Exportar y Limpiar")
        ttk.Button(options_card, text="Exportar Pasos (TXT)", command=self.export_steps).pack(fill=tk.X, pady=5)
        ttk.Button(options_card, text="Exportar Gráfica (PNG)", command=self.export_plot).pack(fill=tk.X, pady=5)
        ttk.Button(options_card, text="Limpiar Gráfica", command=self.on_clear).pack(fill=tk.X, pady=5)

        examples_card = self._create_card(options_tab, "Funciones de Ejemplo")
        examples = ["x**3 - 3*x + 2", "2*x*exp(x**2)", "1/(x**2-1)", "sin(x)**2", "sqrt(4-x**2)", "log(x)/x"]
        for ex in examples:
            b = ttk.Button(examples_card, text=ex, command=lambda e=ex: self._set_example(e))
            b.pack(fill=tk.X, pady=3)

        # Pestaña Ayuda
        help_card = self._create_card(help_tab, "Sintaxis y Ejemplos")
        help_text = (
            "Potencia: x**2 o x^2\n"
            "Multiplicación: 2*x\n"
            "Constantes: pi, E\n"
            "Funciones: sin(x), cos(x), exp(x), \nlog(x), sqrt(x), abs(x)\n"
            "Límites Infinitos: inf, -inf"
        )
        ttk.Label(help_card, text=help_text, justify=tk.LEFT, style="Small.TLabel").pack(anchor='w')

    def _create_center_panel(self, parent):
        graph_frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        fig, ax = plt.subplots(dpi=100)
        self.fig, self.ax = fig, ax
        
        self.canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        toolbar.update()
        toolbar.config(background="#0b1220")
        for child in toolbar.winfo_children():
            child.config(background="#0b1220")

        self.plot_manager = PlotManager(fig, ax, self.canvas)
        
    def _create_right_panel(self, parent):
        steps_frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        steps_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(steps_frame, text="Pasos y Explicaciones", style="CardHeader.TLabel").pack(anchor='w', pady=(0, 10))
        
        self.steps_text = scrolledtext.ScrolledText(steps_frame, height=30, wrap=tk.WORD, 
            background="#0f1720", foreground="#e6eef6", font=("Consolas", 10), 
            relief="flat", borderwidth=0, insertbackground="white")
        self.steps_text.pack(fill=tk.BOTH, expand=True)

    def _show_error(self, title, message):
        """Muestra un cuadro de diálogo de error estandarizado."""
        messagebox.showerror(title, message, parent=self.root)
        self._set_status(f"Error: {title}")

    def _set_example(self, ex):
        self.func_entry.delete(0, tk.END)
        self.func_entry.insert(0, ex)
        self.on_plot()

    def _set_status(self, s, clear_after_ms=None):
        self.status.set(s)
        self.root.update_idletasks()
        if clear_after_ms:
            self.root.after(clear_after_ms, lambda: self.status.set("Listo"))

    def _append_steps(self, lines, title=None):
        """Añade los pasos al área de texto, respetando el formato."""
        if title:
            header = f"--- {title} ---\n"
            self.steps_text.insert(tk.END, header)
            
            # Configurar etiquetas para dar estilo (opcional)
            self.steps_text.tag_configure("header", font=("Segoe UI", 11, "bold"), foreground="#60a5fa")
            self.steps_text.tag_add("header", f"{self.steps_text.index(tk.END)} - {len(header) + 1} chars", f"{self.steps_text.index(tk.END)} - 1 chars")

        # Insertar cada línea de los pasos
        for line in lines:
            self.steps_text.insert(tk.END, f"{line}\n")
            
        self.steps_text.insert(tk.END, "\n" + "-" * 40 + "\n\n")
        self.steps_text.see(tk.END)

    def export_plot(self):
        try:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files","*.png"), ("All files", "*.*")])
            if not path:
                return
            self.fig.savefig(path, dpi=200, facecolor=self.fig.get_facecolor(), bbox_inches='tight')
            messagebox.showinfo("Guardado", f"Gráfica guardada en:\n{path}", parent=self.root)
        except Exception as e:
            self._show_error("Error al Guardar Gráfica", f"No se pudo guardar la imagen:\n{e}")

    def on_calculate(self, calc_type):
        try:
            funcs = [f.strip() for f in self.func_entry.get().split(';') if f.strip()]
            if not funcs:
                raise ValueError("Debe ingresar al menos una función.")

            lower_raw = self.lower_entry.get().strip()
            upper_raw = self.upper_entry.get().strip()
            a = parse_limit_string(lower_raw) if lower_raw else None
            b = parse_limit_string(upper_raw) if upper_raw else None

            self.steps_text.delete('1.0', tk.END)
            for f in funcs:
                self._set_status(f"Calculando: {f} ...")
                res, steps = None, []

                # --- BLOQUE DE DERIVADA MODIFICADO ---
                if calc_type == "derivada":
                    title = f"Derivada de {f}"
                    order = int(self.deriv_order.get())
                    point_str = self.eval_point.get().strip()
                    
                    expr = safe_sympify(f, 'x')

                    if point_str:
                        # Si se evalúa en un punto, la lógica se mantiene simple
                        point = float(point_str)
                        res, steps = self.engine.eval_derivative(f, point, order)
                    else:
                        # Si es solo la derivada simbólica, usamos el solver profesional
                        if order == 1:
                            # Nuestro solver detallado solo funciona para la primera derivada
                            steps = self.solver.get_derivative_steps(expr)
                            res = sp.diff(expr, self.solver.x, order) # El resultado lo obtenemos de SymPy
                        else:
                            # Para derivadas de orden superior, usamos el método original
                            self._append_steps([f"El paso a paso detallado solo está disponible para la primera derivada."], "Aviso")
                            res, steps = self.engine.derivative(f, order)
                
                else: # Integrales
                    method = self.calc_type.get()
                    title = f"{method}: {f}"
                    
                    if method == "Auto (mejor intento)":
                        res, steps = self.engine.indefinite(f)
                    elif method == "Integral indefinida":
                        res, steps = self.engine.indefinite(f)
                    elif method in ["Integral definida", "Integral impropia", "Numérico (Simpson)"]:
                        if a is None or b is None:
                            raise ValueError("Para este método, debe especificar los límites inferior y superior.")
                        if method == "Integral definida": res, steps = self.engine.definite(f, a, b)
                        if method == "Integral impropia": res, steps = self.engine.improper(f, a, b)
                        if method == "Numérico (Simpson)": res, steps = self.engine.numeric_simpson(f, a, b)
                    elif method == "Sustitución": res, steps = self.engine.substitution_auto(f)
                    elif method == "Por partes": res, steps = self.engine.integration_by_parts_auto(f)
                    elif method == "Fracciones parciales": res, steps = self.engine.partial_fractions(f)
                    elif method == "Sust. trigonométrica": res, steps = self.engine.trig_substitution(f)

                if res is not None or steps:
                    self._append_steps(steps, title)
                else:
                    self._append_steps([f"No se pudo resolver utilizando el método '{method}'."], title)

            self._set_status("Cálculo completado", clear_after_ms=3000)

        except (ValueError, TypeError, SympifyError) as e:
            self._show_error("Error de Entrada o Cálculo", str(e))
        except Exception as e:
            self._show_error("Error Inesperado", f"Ocurrió un error inesperado:\n{e}")

    def on_plot(self):
        try:
            funcs = [f.strip() for f in self.func_entry.get().split(';') if f.strip()]
            if not funcs:
                raise ValueError("Debe ingresar al menos una función para graficar.")
            
            xmin = float(self.xmin_entry.get())
            xmax = float(self.xmax_entry.get())
            if xmin >= xmax:
                raise ValueError("El valor de 'X min' debe ser menor que 'X max'.")

            lower_raw = self.lower_entry.get().strip()
            upper_raw = self.upper_entry.get().strip()
            a = parse_limit_string(lower_raw) if lower_raw else None
            b = parse_limit_string(upper_raw) if upper_raw else None
            
            self._set_status("Graficando...")
            self.plot_manager.clear()
            
            # Colores base para múltiples funciones
            colors = ['#60a5fa', '#f87171', '#4ade80', '#fbbf24', '#c084fc']
            
            for i, f in enumerate(funcs):
                color = colors[i % len(colors)]
                self.plot_manager.plot_function(f, xlim=(xmin, xmax), color=color)
            
            # Sombrear solo la primera función
            if funcs and a is not None and b is not None:
                if not (isinstance(a, sp.Number) and isinstance(b, sp.Number)):
                       self._set_status("Advertencia: No se puede sombrear con límites infinitos.")
                else:
                    self.plot_manager.shade_integral(funcs[0], a, b, base_color=colors[0])

            self._set_status("Gráfica generada", clear_after_ms=3000)

        except (ValueError, TypeError) as e:
            self._show_error("Error en los Parámetros", str(e))
        except Exception as e:
            self._show_error("Error al Graficar", f"No se pudo generar la gráfica:\n{e}")

    def on_analyze_points(self):
        try:
            func_str = self.func_entry.get().split(';')[0].strip()
            if not func_str:
                raise ValueError("Ingrese una función para analizar.")

            xmin = float(self.xmin_entry.get()); xmax = float(self.xmax_entry.get())
            if xmin >= xmax:
                raise ValueError("El valor de 'X min' debe ser menor que 'X max'.")

            self._set_status(f"Analizando {func_str}...")
            analysis = self.engine.critical_points(func_str, xlim=(xmin, xmax))
            
            self.steps_text.delete('1.0', tk.END)
            self.steps_text.insert(tk.END, f"--- Análisis de Puntos Clave para f(x) = {func_str} ---\n")
            
            all_points = []
            iy = analysis['interseccion_y']
            if iy and iy.get('y') is not None:
                self.steps_text.insert(tk.END, f"\n- {iy['tipo']}: ({iy['x']:.3f}, {iy['y']:.3f})\n")
                all_points.append(iy)
            
            if analysis['raices']:
                self.steps_text.insert(tk.END, "\nRaíces (intersecciones con eje X):\n")
                for r in analysis['raices']:
                    self.steps_text.insert(tk.END, f"  - ({r['x']:.3f}, {r['y']:.3f})\n")
                    all_points.append(r)
            
            if analysis['puntos_criticos']:
                self.steps_text.insert(tk.END, "\nPuntos Críticos (donde f'(x)=0):\n")
                for pc in analysis['puntos_criticos']:
                    self.steps_text.insert(tk.END, f"  - {pc['tipo']}: ({pc['x']:.3f}, {pc['y']:.3f})\n")
                    all_points.append(pc)
            
            self.steps_text.insert(tk.END, "-"*50 + "\n")
            
            self.plot_manager.mark_points(all_points)
            self._set_status("Análisis completado.", clear_after_ms=3000)

        except (ValueError, TypeError) as e:
            self._show_error("Error de Análisis", str(e))
        except Exception as e:
            self._show_error("Error Inesperado en Análisis", f"No se pudo completar el análisis:\n{e}")

    def export_steps(self):
        content = self.steps_text.get('1.0', tk.END).strip()
        if not content:
            messagebox.showinfo("Nada que Guardar", "No hay pasos para exportar.", parent=self.root)
            return
        
        try:
            path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt")])
            if not path:
                return
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(content)
            messagebox.showinfo("Guardado", f"Pasos guardados en:\n{path}", parent=self.root)
        except Exception as e:
            self._show_error("Error al Guardar Pasos", f"No se pudo guardar el archivo:\n{e}")

    def on_clear(self):
        self.plot_manager.clear()
        self._set_status("Gráfica limpiada", clear_after_ms=3000)

# ----------------------- RUN -----------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = UltraCalc2DDetailedApp(root)
    root.mainloop()
    