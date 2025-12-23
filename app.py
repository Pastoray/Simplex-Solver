from flask import Flask, render_template, request
import numpy as np
from collections import defaultdict
import re


def parse_objective(obj_str):
    obj_str = obj_str.strip()
    if obj_str.lower().startswith('max '):
        is_max = True
        expr = obj_str[4:].strip()
    elif obj_str.lower().startswith('min '):
        is_max = False
        expr = obj_str[4:].strip()
    else:
        raise ValueError("Objective must start with 'min' or 'max'")

    var_pattern = r'([+-]?\s*\d*\.?\d*\*?)?([xX]\d+)'
    terms = re.findall(var_pattern, expr.replace(' ', ''))

    if not terms:
        raise ValueError("No variables found in objective")

    coeffs = defaultdict(float)
    for coef_str, var in terms:
        var = var.lower()
        coef_str = coef_str.strip()
        if coef_str == '' or coef_str == '+':
            coef = 1.0
        elif coef_str == '-':
            coef = -1.0
        else:
            coef_str = coef_str.replace('*', '')
            if coef_str in ('+', ''):
                coef = 1.0
            elif coef_str == '-':
                coef = -1.0
            else:
                try:
                    coef = float(coef_str)
                except ValueError:
                    coef = 1.0 if coef_str.endswith('+') else -1.0
        coeffs[var] += coef

    var_names = sorted(coeffs.keys(), key=lambda x: int(x[1:]))
    c = [coeffs[var] for var in var_names]

    if not is_max:
        c = [-x for x in c]

    return is_max, c, var_names


def parse_constraints(constraints_str, var_names):
    lines = [line.strip() for line in constraints_str.strip().splitlines() if line.strip()]
    A = []
    b = []
    senses = []
    var_set = set(var_names)
    var_pattern = r'([+-]?\s*\d*\.?\d*\*?)?([xX]\d+)'

    for line in lines:
        if '<=' in line:
            sense = '<='
            parts = line.split('<=')
        elif '>=' in line:
            sense = '>='
            parts = line.split('>=')
        elif '=' in line:
            sense = '='
            parts = line.split('=')
        else:
            raise ValueError(f"Constraint must contain <=, >=, or = : {line}")

        if len(parts) != 2:
            raise ValueError(f"Invalid constraint: {line}")

        lhs = parts[0].strip()
        rhs = float(parts[1].strip())
        terms = re.findall(var_pattern, lhs.replace(' ', ''))
        row = [0.0] * len(var_names)

        for coef_str, var in terms:
            var = var.lower()
            if var not in var_set:
                raise ValueError(f"Unknown variable in constraint: {var}")
            idx = var_names.index(var)
            coef_str = coef_str.strip()
            if coef_str == '' or coef_str == '+':
                coef = 1.0
            elif coef_str == '-':
                coef = -1.0
            else:
                coef_str = coef_str.replace('*', '')
                try:
                    coef = float(coef_str)
                except ValueError:
                    coef = 1.0
            row[idx] += coef

        A.append(row)
        b.append(rhs)
        senses.append(sense)

    return A, b, senses


class SimplexSolver:
    def __init__(self, c, A, b, senses, method='regular', M=1e6, var_names=None, is_max=False):
        self.c_orig = np.array(c, dtype=float)
        self.A_orig = np.array(A, dtype=float)
        self.b_orig = np.array(b, dtype=float)
        self.is_max = is_max
        self.senses = senses
        self.method = method
        self.M = M
        self.m, self.n = self.A_orig.shape
        self.user_var_names = var_names if var_names else [f"x{i + 1}" for i in range(self.n)]
        self.output_log = []
        self._construct_tableau()

    def _construct_tableau(self):
        A = self.A_orig.copy()
        b = self.b_orig.copy()
        c = self.c_orig.copy()
        m, n = A.shape
        senses = self.senses

        slack_count = 0
        surplus_count = 0
        artificial_count = 0

        for sense in senses:
            if sense == '<=':
                slack_count += 1
            elif sense == '>=':
                surplus_count += 1
                artificial_count += 1
            elif sense == '=':
                artificial_count += 1

        total_extra = slack_count + surplus_count + artificial_count
        tableau_rows = m
        tableau_cols = n + total_extra + 1

        tableau = np.zeros((tableau_rows, tableau_cols))
        tableau[:, :n] = A
        tableau[:, -1] = b

        var_names = self.user_var_names.copy()
        c_full = c.tolist()

        basic_vars = []
        current_col = n

        slack_idx = 0
        surplus_idx = 0
        artificial_idx = 0

        for i, sense in enumerate(senses):
            if sense == '<=':
                tableau[i, current_col] = 1.0
                var_names.append(f"s{slack_idx + 1}")
                c_full.append(0.0)
                basic_vars.append(current_col)
                slack_idx += 1
                current_col += 1
            elif sense == '>=':
                tableau[i, current_col] = -1.0
                var_names.append(f"sur{surplus_idx + 1}")
                c_full.append(0.0)
                surplus_idx += 1
                current_col += 1

                tableau[i, current_col] = 1.0
                var_names.append(f"a{artificial_idx + 1}")
                if self.method == 'big_m':
                    c_full.append(-self.M)
                else:
                    c_full.append(0.0)
                basic_vars.append(current_col)
                artificial_idx += 1
                current_col += 1
            elif sense == '=':
                tableau[i, current_col] = 1.0
                var_names.append(f"a{artificial_idx + 1}")
                if self.method == 'big_m':
                    c_full.append(-self.M)
                else:
                    c_full.append(0.0)
                basic_vars.append(current_col)
                artificial_idx += 1
                current_col += 1

        self.tableau = tableau
        self.basic_vars = basic_vars
        self.var_names = var_names
        self.c_full = np.array(c_full)

        if self.method == 'two_phase':
            self.c_full_phase1 = np.zeros(len(c_full))
            art_indices = [i for i, name in enumerate(var_names) if name.startswith('a')]
            for idx in art_indices:
                self.c_full_phase1[idx] = 1.0
            self.c_full_phase2 = self.c_full.copy()

    def _log(self, html_content):
        self.output_log.append(html_content)

    def _render_tableau(self, phase="", entering_col=None, leaving_row=None, note=""):
        m, total_cols = self.tableau.shape
        n_vars = total_cols - 1

        Cj_full = self.c_full.copy()
        Cj_display = np.concatenate([Cj_full, [0.0]])

        Cb = np.array([self.c_full[i] for i in self.basic_vars])
        Zj = Cb @ self.tableau

        Cj_minus_Zj = Cj_display.copy()
        Cj_minus_Zj[:n_vars] = Cj_full - Zj[:n_vars]
        Cj_minus_Zj[-1] = 0.0

        html = f'<h3>{phase}</h3>'
        if note:
            html += f'<p><em>{note}</em></p>'
        html += '<table class="tableau">'

        html += '<tr><th>Cj</th>'
        for j in range(n_vars):
            html += f'<th>{self.c_full[j]:.3f}</th>'
        html += '<th></th>'
        html += '</tr>'

        headers = self.var_names + ["b"]
        html += '<tr><th>Basic Var</th>'
        for h in headers:
            html += f'<th>{h}</th>'
        html += '</tr>'

        for i in range(m):
            basic_name = self.var_names[self.basic_vars[i]]
            html += f'<tr><td>{basic_name}</td>'
            row = self.tableau[i, :]
            for j, val in enumerate(row):
                cell_class = ""
                marker = ""
                if entering_col is not None and leaving_row is not None and j == entering_col and i == leaving_row:
                    marker = ""
                    cell_class = "pivot-elem"
                elif entering_col is not None and j == entering_col:
                    marker = ""
                    cell_class = "pivot-col"
                elif leaving_row is not None and i == leaving_row:
                    marker = ""
                    cell_class += " pivot-row"
                formatted_val = f"{val:.3f}{marker}"
                html += f'<td class="{cell_class}">{formatted_val}</td>'
            html += '</tr>'

        html += '<tr><td>Zj</td>'
        for j in range(n_vars + 1):
            cell_class = "pivot-col" if (entering_col is not None and j == entering_col) else ""
            html += f'<td class="{cell_class}">{Zj[j]:.3f}</td>'
        html += '</tr>'

        html += '<tr><td>Cj - Zj</td>'
        for j in range(n_vars):
            val = self.c_full[j] - Zj[j]
            cell_class = ""
            if entering_col is not None and j == entering_col:
                cell_class = "pivot-col"
            html += f'<td class="{cell_class}">{val:.3f}</td>'
        html += '<td></td>'
        html += '</tr>'

        sol = np.zeros(len(self.var_names))
        for idx, bv in enumerate(self.basic_vars):
            sol[bv] = self.tableau[idx, -1]
        obj_val = np.dot(self.c_orig, sol[:len(self.c_orig)])
        if not self.is_max:
            obj_val = -obj_val

        sol_str = ", ".join([f"{self.var_names[i]}={sol[i]:.3f}" for i in range(len(self.c_orig))])
        html += f'<p><strong>Objective:</strong> {obj_val:.3f} | <strong>Solution:</strong> {sol_str}</p>'
        return html

    def _print_tableau(self, phase="", entering_col=None, leaving_row=None, note=""):
        html = self._render_tableau(phase, entering_col, leaving_row, note)
        self._log(html)

    def _pivot(self, pivot_row, pivot_col):
        pivot_val = self.tableau[pivot_row, pivot_col]
        if abs(pivot_val) < 1e-10:
            raise ValueError("Pivot element is zero")
        self.tableau[pivot_row, :] /= pivot_val
        for i in range(self.tableau.shape[0]):
            if i != pivot_row:
                factor = self.tableau[i, pivot_col]
                self.tableau[i, :] -= factor * self.tableau[pivot_row, :]
        self.basic_vars[pivot_row] = pivot_col

    def regular_simplex(self):
        self._log("<h2>Regular Simplex Method</h2>")
        self._print_tableau(phase="Initial Tableau", note="Starting state")
        iteration = 0
        while True:
            iteration += 1
            c_B = np.array([self.c_full[i] for i in self.basic_vars])
            obj_row = c_B @ self.tableau[:, :-1] - self.c_full
            if np.all(obj_row >= -1e-10):
                self._print_tableau(phase="Optimal Tableau", note="Optimal solution found.")
                break

            entering = np.argmin(obj_row)
            col = self.tableau[:, entering]
            rhs = self.tableau[:, -1]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.where(col > 1e-10, rhs / col, np.inf)
            if np.all(ratios == np.inf):
                self._log("<p><strong>Unbounded solution</strong></p>")
                return

            leaving = np.argmin(ratios)
            self._print_tableau(
                phase=f"Iteration {iteration} - Pivot Selection",
                entering_col=entering,
                leaving_row=leaving,
                note=f"Entering: {self.var_names[entering]}, Leaving: {self.var_names[self.basic_vars[leaving]]}"
            )
            self._pivot(leaving, entering)
            self._print_tableau(phase=f"Iteration {iteration} - After Pivot")

    def two_phase_simplex(self):
        self._log("<h2>Two-Phase Simplex Method</h2>")
        self._log("<h3>Phase 1: Minimize sum of artificial variables</h3>")
        self.c_full = self.c_full_phase1.copy()
        self._print_tableau(phase="Phase 1 - Initial", note="Artificial variables in basis")

        iteration = 0
        while True:
            iteration += 1
            c_B = np.array([self.c_full[i] for i in self.basic_vars])
            obj_row = c_B @ self.tableau[:, :-1] - self.c_full
            if np.all(obj_row >= -1e-10):
                break
            entering = np.argmin(obj_row)
            col = self.tableau[:, entering]
            rhs = self.tableau[:, -1]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.where(col > 1e-10, rhs / col, np.inf)
            if np.all(ratios == np.inf):
                self._log("<p>Phase 1 unbounded - infeasible.</p>")
                return
            leaving = np.argmin(ratios)
            self._print_tableau(
                phase=f"Phase 1 - Iter {iteration} (Pivot)",
                entering_col=entering,
                leaving_row=leaving,
                note=f"Entering {self.var_names[entering]}, Leaving {self.var_names[self.basic_vars[leaving]]}"
            )
            self._pivot(leaving, entering)
            self._print_tableau(phase=f"Phase 1 - Iter {iteration} (After)")

        phase1_obj = np.array([self.c_full[i] for i in self.basic_vars]) @ self.tableau[:, -1]
        if abs(phase1_obj) > 1e-5:
            self._log(f"<p>Infeasible: Phase 1 objective = {phase1_obj:.3f} > 0</p>")
            return

        self._log("<h3>Phase 2: Optimize original objective</h3>")
        self.c_full = self.c_full_phase2.copy()
        n_orig = len(self.c_orig)
        extra_vars = len(self.var_names) - n_orig

        for i, bv in enumerate(self.basic_vars):
            if bv >= n_orig and self.var_names[bv].startswith('a'):
                for j in range(n_orig):
                    if abs(self.tableau[i, j]) > 1e-8:
                        self._pivot(i, j)
                        break

        self._print_tableau(phase="Phase 2 - Initial", note="Artificials removed (if possible)")

        iteration = 0
        while True:
            iteration += 1
            c_B = np.array([self.c_full[i] for i in self.basic_vars])
            obj_row = c_B @ self.tableau[:, :-1] - self.c_full
            if np.all(obj_row >= -1e-10):
                self._print_tableau(phase="Optimal Tableau (Phase 2)", note="Optimal solution found.")
                break
            entering = np.argmin(obj_row)
            col = self.tableau[:, entering]
            rhs = self.tableau[:, -1]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.where(col > 1e-10, rhs / col, np.inf)
            if np.all(ratios == np.inf):
                self._log("<p>Unbounded.</p>")
                return
            leaving = np.argmin(ratios)
            self._print_tableau(
                phase=f"Phase 2 - Iter {iteration} (Pivot)",
                entering_col=entering,
                leaving_row=leaving,
                note=f"Entering {self.var_names[entering]}, Leaving {self.var_names[self.basic_vars[leaving]]}"
            )
            self._pivot(leaving, entering)
            self._print_tableau(phase=f"Phase 2 - Iter {iteration} (After)")

    def solve(self):
        try:
            if self.method == 'regular':
                for sense in self.senses:
                    if sense != '<=':
                        raise ValueError("Regular simplex only supports '<=' constraints.")
                self.regular_simplex()
            elif self.method == 'two_phase':
                self.two_phase_simplex()
            else:
                raise ValueError("Unknown method")

            sol = np.zeros(len(self.var_names))
            for i, bv in enumerate(self.basic_vars):
                sol[bv] = self.tableau[i, -1]
            primal = sol[:len(self.c_orig)]
            obj = np.dot(self.c_orig, primal)
            if not self.is_max:
                obj = -obj

            final = f"""
            <div class="final-solution">
                <h3>Final Solution</h3>
                <p><strong>Optimal Objective Value:</strong> {obj:.3f}</p>
                <p><strong>Variable Values:</strong><br>
                {'<br>'.join([f"{self.var_names[i]} = {primal[i]:.3f}" for i in range(len(primal))])}
                </p>
            </div>
            """
            self._log(final)
            return "\n".join(self.output_log)
        except Exception as e:
            return f"<p>Error: {str(e)}</p>"


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        try:
            objective_str = request.form['objective']
            constraints_str = request.form['constraints']
            method = request.form['method']
            M = float(request.form.get('M', 100000))

            is_max, c, var_names = parse_objective(objective_str)

            A, b, senses = parse_constraints(constraints_str, var_names)

            for i in range(len(A)):
                if b[i] < 0:
                    A[i] = [-x for x in A[i]]
                    b[i] = -b[i]
                    if senses[i] == '<=':
                        senses[i] = '>='
                    elif senses[i] == '>=':
                        senses[i] = '<='

            if method == 'regular':
                for i, sense in enumerate(senses):
                    if sense != '<=':
                        if sense == '>=':
                            A[i] = [-x for x in A[i]]
                            b[i] = -b[i]
                            senses[i] = '<='
                        else:
                            raise ValueError(
                                "Regular simplex only supports '<=' or '>=' constraints. Use Two-Phase or Big-M for other constraints.")

            solver = SimplexSolver(c, A, b, senses, method=method, M=M, var_names=var_names, is_max=is_max)
            result = solver.solve()
        except Exception as e:
            result = f"<p>Input error: {str(e)}</p>"

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)