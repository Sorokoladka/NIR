"""
quality/calibration.py

Научно обоснованный модуль калибровки и верификации стохастической модели NIR.

Применяемые методы:
  1. GBM акций          — метод моментов для log-доходности: t-тест (E) и χ²-тест (Var)
  2. Vasicek облигации  — аналитические моменты OU-процесса: E[Y_T] и Var[Y_T]
  3. Корреляции портфеля — сохранение структуры через разложение Холецкого
  4. Детерм. зарплата   — точная формула сложного роста (машинная точность)
  5. Безработица: частота   — биномиальный тест для p_exit
  6. Безработица: длительность — критерий Колмогорова–Смирнова против Weibull(k, λ)
  7. Аннуитетная формула — FV калькулятора vs. аналитическая FV аннуитета-дью
  8. Монотонность        — взносы / комиссии / налог / софинансирование
  9. Сходимость МК       — std_error убывает ∝ 1/√N (ЦПТ)
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass, field

from models.securities.stocks import StockModel, StockParams
from models.securities.bonds import BondModel, BondParams
from models.macro.salary import DeterministicSalaryModel
from models.macro.unemployment import WeibullUnemploymentModel
from models.programs.base import ProgramInput
from models.programs.iis3 import IIS3Program
from models.programs.pds import PDSProgram


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        return f"[{tag}] {self.name}: {self.message}"


# ══════════════════════════════════════════════════════════════════════════════
# Calibration suite
# ══════════════════════════════════════════════════════════════════════════════

class CalibrationSuite:
    """
    Запускает 9 групп тестов, покрывающих все компоненты модели.

    Parameters
    ----------
    n_sims : число реплик Монте-Карло для стохастических тестов
    alpha  : уровень значимости для статистических критериев
    seed   : seed для воспроизводимости
    """

    def __init__(self, n_sims: int = 3000, alpha: float = 0.05, seed: int = 42):
        self.n_sims = n_sims
        self.alpha = alpha
        self.seed = seed
        self.results: list[TestResult] = []

    # ── public API ────────────────────────────────────────────────────────────

    def run_all(self) -> list[TestResult]:
        """Запустить все тесты и вернуть список результатов."""
        np.random.seed(self.seed)
        self.results = []

        self._test_stock_log_moments()
        self._test_vasicek_moments()
        self._test_portfolio_correlation()
        self._test_salary_deterministic()
        self._test_unemployment_frequency()
        self._test_unemployment_duration_ks()
        self._test_program_annuity_formula()
        self._test_program_monotonicity()
        self._test_monte_carlo_convergence()

        return self.results

    def summary(self) -> str:
        """Сформировать текстовый отчёт по всем тестам."""
        passed = sum(r.passed for r in self.results)
        total = len(self.results)
        lines = [
            "=" * 65,
            f"  Calibration Report  —  {passed}/{total} tests passed",
            "=" * 65,
        ]
        for r in self.results:
            lines.append(str(r))
        lines.append("=" * 65)
        return "\n".join(lines)

    def _add(self, result: TestResult) -> None:
        self.results.append(result)

    # ── 1. Stock: GBM log-moment test ─────────────────────────────────────────

    def _test_stock_log_moments(
        self,
        mu: float = 0.07,
        sigma: float = 0.15,
        dt: float = 1 / 252,
    ) -> None:
        """
        Проверяет, что годовые log-доходности акций следуют N(μ−σ²/2, σ²).

        Для дискретной аппроксимации GBM с шагом dt:
            r[t] = μ·dt + σ·√dt·Z[t]

        Скользящее произведение 252 дневных шагов (rolling product) аппроксимирует
        непрерывный GBM, и по лемме Ито:
            ln(1 + R_annual) ~ N(μ − σ²/2, σ²)

        Тесты: двусторонний t-тест для E[ln(1+R)] и χ²-тест для Var[ln(1+R)].
        """
        name = "GBM акций: моменты log-доходности"
        n_steps = int(1.0 / dt)  # 252 шага = 1 год
        model = StockModel(StockParams(weight=1.0, mu=mu, sigma=sigma))

        log_returns = np.empty(self.n_sims)
        for i in range(self.n_sims):
            Z = np.random.normal(size=n_steps)
            path = model.simulate_path(Z, n_years=1.0, dt=dt)
            log_returns[i] = np.log(1.0 + path[-1])  # последний элемент — первое полное окно

        finite = log_returns[np.isfinite(log_returns)]
        n = len(finite)

        theo_mean = mu - sigma ** 2 / 2.0
        theo_var = sigma ** 2

        # t-тест для среднего
        _, p_mean = stats.ttest_1samp(finite, popmean=theo_mean)

        # χ²-тест для дисперсии
        chi2 = (n - 1) * np.var(finite, ddof=1) / theo_var
        p_var = 2 * min(stats.chi2.cdf(chi2, n - 1), stats.chi2.sf(chi2, n - 1))

        rel_err_mean = abs(finite.mean() - theo_mean) / (abs(theo_mean) + 1e-12)
        rel_err_std = abs(np.std(finite, ddof=1) - sigma) / sigma
        passed = (p_mean > self.alpha) and (p_var > self.alpha)

        self._add(TestResult(
            name=name, passed=passed,
            message=(f"err_mean={rel_err_mean:.3%}, err_std={rel_err_std:.3%}, "
                     f"p_mean={p_mean:.3f}, p_var={p_var:.3f}"),
            details=dict(
                theoretical_mean=theo_mean, simulated_mean=float(finite.mean()),
                theoretical_std=sigma, simulated_std=float(np.std(finite, ddof=1)),
                p_mean=float(p_mean), p_var=float(p_var),
            ),
        ))

    # ── 2. Bond: Vasicek moments ──────────────────────────────────────────────

    def _test_vasicek_moments(
        self,
        a: float = 0.1,
        b: float = 0.07,
        sigma: float = 0.015,
        Y0: float = 0.05,
        horizons: tuple[int, ...] = (1, 5, 10),
        rtol: float = 0.05,
        dt: float = 1 / 252,
    ) -> None:
        """
        Проверяет аналитические моменты процесса Васичека (OU) на нескольких горизонтах.

        Аналитические выражения для процесса dY = a(b−Y)dt + σdW:
            E[Y_T]   = b + (Y₀ − b)·exp(−aT)
            Var[Y_T] = σ²/(2a)·(1 − exp(−2aT))

        Тест: относительная ошибка симуляции vs. теории < rtol на каждом горизонте.
        """
        name = "Vasicek облигации: сходимость моментов"
        max_T = max(horizons)
        n_steps = int(max_T / dt)
        model = BondModel(BondParams(weight=1.0, sigma=sigma, a=a, b=b, Y0=Y0))

        # симулируем n_sims путей длиной max_T
        last_indices = {T: int(T / dt) - 1 for T in horizons}
        snapshots = {T: np.empty(self.n_sims) for T in horizons}

        for i in range(self.n_sims):
            Z = np.random.normal(size=n_steps)
            path = model.simulate_path(Z, n_years=max_T, dt=dt)
            for T in horizons:
                snapshots[T][i] = path[last_indices[T]]

        errors: dict[str, dict] = {}
        all_passed = True

        for T in horizons:
            vals = snapshots[T]
            theo_mean = b + (Y0 - b) * np.exp(-a * T)
            theo_var = (sigma ** 2 / (2 * a)) * (1 - np.exp(-2 * a * T))

            err_mean = abs(vals.mean() - theo_mean) / (abs(theo_mean) + 1e-12)
            err_var = abs(vals.var(ddof=1) - theo_var) / (abs(theo_var) + 1e-12)

            errors[f"T={T}"] = dict(
                theo_mean=round(theo_mean, 6), sim_mean=round(float(vals.mean()), 6),
                theo_var=round(theo_var, 8), sim_var=round(float(vals.var(ddof=1)), 8),
                rel_err_mean=err_mean, rel_err_var=err_var,
            )
            if err_mean > rtol or err_var > rtol:
                all_passed = False

        worst_mean = max(v["rel_err_mean"] for v in errors.values())
        worst_var = max(v["rel_err_var"] for v in errors.values())
        self._add(TestResult(
            name=name, passed=all_passed,
            message=(f"max_err_mean={worst_mean:.3%}, max_err_var={worst_var:.3%} "
                     f"(rtol={rtol:.1%})"),
            details=errors,
        ))

    # ── 3. Portfolio: Cholesky correlation preservation ───────────────────────

    def _test_portfolio_correlation(
        self,
        rho: float = 0.4,
        n_steps: int = 252,
        atol: float = 0.04,
    ) -> None:
        """
        Проверяет, что разложение Холецкого правильно транслирует корреляции.

        Портфель использует Z_corr = L @ Z, L = chol(Σ).
        По свойству разложения: Cov(Z_corr) = L·L^T = Σ.

        Тест: |ρ_sample − ρ_target| < atol.
        Для n_sims=3000 стандартная ошибка коэф. корр. ≈ (1−ρ²)/√n ≈ 0.006,
        поэтому atol=0.04 соответствует ~6σ-порогу.
        """
        name = "Портфель: сохранение корреляций (Холецкий)"
        corr_target = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(corr_target)

        # одна случайная величина на актив × симуляция = среднее по шагам
        shocks = np.empty((self.n_sims, 2))
        for i in range(self.n_sims):
            Z = np.random.normal(size=(2, n_steps))
            Z_corr = L @ Z
            shocks[i] = Z_corr.mean(axis=1)

        sample_rho = float(np.corrcoef(shocks.T)[0, 1])
        err = abs(sample_rho - rho)
        passed = err < atol

        self._add(TestResult(
            name=name, passed=passed,
            message=f"ρ_target={rho:.3f}, ρ_sample={sample_rho:.4f}, |err|={err:.4f} (atol={atol})",
            details=dict(target_rho=rho, sample_rho=sample_rho, abs_error=err),
        ))

    # ── 4. Salary: exact deterministic formula ────────────────────────────────

    def _test_salary_deterministic(
        self,
        initial_salary: float = 60_000.0,
        annual_growth: float = 0.05,
        n_years: int = 20,
        atol: float = 1e-6,
    ) -> None:
        """
        Проверяет, что DeterministicSalaryModel точно воспроизводит формулу:
            salary_annual[t] = initial_salary · (1+g)^t · 12

        Тест: max_t |result[t] − expected[t]| < atol (машинная точность).
        """
        name = "Зарплата: детерминированная формула"
        model = DeterministicSalaryModel(annual_growth=annual_growth)
        result = model.simulate(n_years=n_years, initial_salary=initial_salary)

        expected = np.array(
            [initial_salary * (1.0 + annual_growth) ** t * 12 for t in range(n_years)]
            + [0.0]
        )
        max_err = float(np.max(np.abs(result - expected)))
        passed = max_err < atol

        self._add(TestResult(
            name=name, passed=passed,
            message=f"max_abs_err={max_err:.2e} (atol={atol:.0e})",
            details=dict(max_abs_error=max_err),
        ))

    # ── 5. Unemployment: event frequency ──────────────────────────────────────

    def _test_unemployment_frequency(
        self,
        p_exit: float = 0.05,
        weibull_k: float = 1.5,
        weibull_lambda: float = 6.0,
        n_years: int = 20,
    ) -> None:
        """
        Биномиальный тест: доля лет с безработицей ≈ 1 − p_exit.

        В WeibullUnemploymentModel каждый год — испытание Бернулли:
            P(занят)    = p_exit
            P(безработица) = 1 − p_exit

        Суммарное число «шоковых» лет ~ Binomial(n_sims · n_years, 1 − p_exit).
        Тест использует нормальное приближение биномиального (ЦПТ).
        """
        name = "Безработица: частота событий (биномиальный тест)"
        model = WeibullUnemploymentModel(
            p_exit=p_exit, weibull_k=weibull_k, weibull_lambda=weibull_lambda
        )

        total_shocks = 0
        for _ in range(self.n_sims):
            shocks = model.simulate_shocks(n_years)
            # shocks[0] = 1.0 (инициализация), shocks[1:] — результаты испытаний
            # binary=0 (shock) → shocks[i] = employment < 1.0 (Weibull duration > 0)
            # binary=1 (employed) → shocks[i] = 1.0
            total_shocks += int(np.sum(shocks[1:] < 1.0))

        n_total = self.n_sims * n_years
        # P(shocks[i] < 1) = P(binary = 0) = p_exit (вероятность потери работы)
        p_expected = p_exit
        p_hat = total_shocks / n_total

        # z-статистика нормального приближения
        z = (p_hat - p_expected) / np.sqrt(p_expected * (1 - p_expected) / n_total)
        p_value = float(2 * stats.norm.sf(abs(z)))
        passed = p_value > self.alpha

        self._add(TestResult(
            name=name, passed=passed,
            message=(f"p_shock: ожид.={p_expected:.4f}, набл.={p_hat:.4f}, "
                     f"z={z:.3f}, p={p_value:.3f}"),
            details=dict(p_expected=p_expected, p_observed=p_hat, z_stat=z, p_value=p_value),
        ))

    # ── 6. Unemployment: duration KS test ─────────────────────────────────────

    def _test_unemployment_duration_ks(
        self,
        p_exit: float = 0.05,
        weibull_k: float = 1.5,
        weibull_lambda: float = 6.0,
    ) -> None:
        """
        Тест Колмогорова–Смирнова: длительности безработицы ~ Weibull(k, λ).

        Модель использует метод обратной функции:
            D = λ · (−ln U)^{1/k}, U ~ U(0, 1)

        Что эквивалентно выборке из Weibull(shape=k, scale=λ).
        Тест проверяет сырые длительности (до деления на 12 и обрезки).
        """
        name = "Безработица: распределение длительности (KS vs. Weibull)"
        model = WeibullUnemploymentModel(
            p_exit=p_exit, weibull_k=weibull_k, weibull_lambda=weibull_lambda
        )

        sample = model._generate_duration(self.n_sims)
        ks_stat, p_value = stats.kstest(
            sample,
            "weibull_min",
            args=(weibull_k, 0, weibull_lambda),
        )
        passed = float(p_value) > self.alpha

        self._add(TestResult(
            name=name, passed=passed,
            message=f"KS={ks_stat:.4f}, p={p_value:.4f} (α={self.alpha})",
            details=dict(ks_statistic=float(ks_stat), p_value=float(p_value)),
        ))

    # ── 7. Program: annuity-due formula ──────────────────────────────────────

    def _test_program_annuity_formula(
        self,
        c: float = 10_000.0,
        r: float = 0.08,
        n: int = 15,
        rtol: float = 1e-9,
    ) -> None:
        """
        Аналитическая верификация калькулятора против формулы аннуитета-дью.

        При постоянном взносе c, постоянной доходности r, нулевых комиссиях,
        нулевом налоге и нулевом софинансировании, взнос происходит в начале
        каждого периода (annuity-due):

            FV = c · (1+r) · [(1+r)^n − 1] / r

        Вывод из трассировки модели:
            balance_after_fee[k] = c·(1+r) + c·(1+r)² + … + c·(1+r)^{k+1}

        Тест: |FV_model − FV_analytical| / FV_analytical < rtol.
        """
        name = "Калькулятор: аналитическая формула аннуитета (IIS3)"
        params = ProgramInput(
            n=n, age=30, sex="M",
            rates=[r] * n,
            payment_mode="const",
            fix_payment=c,
            initial_salary=50_000.0,
            tax_deduction_rate=0.0,
        )
        program = IIS3Program(params=params, life_table=None)
        program.run()

        fv_model = program.final_accumulation
        fv_analytical = c * (1.0 + r) * ((1.0 + r) ** n - 1.0) / r
        rel_err = abs(fv_model - fv_analytical) / fv_analytical
        passed = rel_err < rtol

        self._add(TestResult(
            name=name, passed=passed,
            message=(f"FV_model={fv_model:,.2f}, FV_analytical={fv_analytical:,.2f}, "
                     f"rel_err={rel_err:.2e} (rtol={rtol:.0e})"),
            details=dict(fv_model=fv_model, fv_analytical=fv_analytical, rel_error=rel_err),
        ))

    # ── 8. Program: monotonicity checks ──────────────────────────────────────

    def _test_program_monotonicity(
        self,
        r: float = 0.08,
        n: int = 15,
    ) -> None:
        """
        Экономические свойства монотонности модели:

        (a) Пропорциональность: FV(2c) = 2·FV(c) для нулевых комиссий (IIS3)
            — проверяет линейность калькулятора по взносам.
        (b) Монотонность взноса: FV(5k) < FV(10k) < FV(20k) (IIS3).
        (c) Налоговый вычет: FV(tax=13%) > FV(tax=0%) — вычет увеличивает накопление.
        (d) Софинансирование: FV_PDS (зарплата < 80k) > FV_IIS3 — государственный
            мэтчинг перекрывает комиссии ПДС при низкой зарплате.
        """
        name = "Монотонность программы (4 проверки)"

        def iis3(fix_payment, tax=0.0, salary=50_000.0):
            p = ProgramInput(
                n=n, age=30, sex="M", rates=[r] * n,
                payment_mode="const", fix_payment=fix_payment,
                initial_salary=salary, tax_deduction_rate=tax,
            )
            prog = IIS3Program(params=p, life_table=None)
            prog.run()
            return prog.final_accumulation

        def pds(fix_payment, salary=50_000.0, tax=0.0):
            p = ProgramInput(
                n=n, age=30, sex="M", rates=[r] * n,
                payment_mode="const", fix_payment=fix_payment,
                initial_salary=salary, tax_deduction_rate=tax,
            )
            prog = PDSProgram(params=p, life_table=None)
            prog.run()
            return prog.final_accumulation

        checks = {}

        # (a) Линейность по взносу: FV(2c) = 2·FV(c)
        fv_c = iis3(10_000)
        fv_2c = iis3(20_000)
        ratio = fv_2c / fv_c
        checks["(a) линейность FV(2c)=2·FV(c)"] = dict(
            fv_c=round(fv_c, 2), fv_2c=round(fv_2c, 2),
            ratio=round(ratio, 8), passed=abs(ratio - 2.0) < 1e-9,
        )

        # (b) Монотонность: FV(5k) < FV(10k) < FV(20k)
        fv_5, fv_10, fv_20 = iis3(5_000), iis3(10_000), iis3(20_000)
        checks["(b) монотонность взноса"] = dict(
            fv_5k=round(fv_5, 2), fv_10k=round(fv_10, 2), fv_20k=round(fv_20, 2),
            passed=bool(fv_5 < fv_10 < fv_20),
        )

        # (c) Налоговый вычет: tax=13% даёт больше, чем tax=0%
        fv_no_tax = iis3(10_000, tax=0.0)
        fv_tax = iis3(10_000, tax=0.13)
        checks["(c) налоговый вычет увеличивает FV"] = dict(
            fv_no_tax=round(fv_no_tax, 2), fv_tax_13pct=round(fv_tax, 2),
            passed=bool(fv_tax > fv_no_tax),
        )

        # (d) Государственный мэтчинг ПДС > IIS3 при низкой зарплате (без налога)
        fv_iis = iis3(10_000, salary=50_000.0, tax=0.0)
        fv_pds = pds(10_000, salary=50_000.0, tax=0.0)
        checks["(d) ПДС(co-fin) > IIS3 при z<80k"] = dict(
            fv_iis3=round(fv_iis, 2), fv_pds=round(fv_pds, 2),
            passed=bool(fv_pds > fv_iis),
        )

        all_passed = all(v["passed"] for v in checks.values())
        failed = [k for k, v in checks.items() if not v["passed"]]
        msg = "все проверки пройдены" if all_passed else f"не пройдены: {failed}"

        self._add(TestResult(
            name=name, passed=all_passed, message=msg, details=checks,
        ))

    # ── 9. Monte Carlo: CLT convergence test ──────────────────────────────────

    def _test_monte_carlo_convergence(
        self,
        r_mean: float = 0.08,
        r_sigma: float = 0.10,
        n_years: int = 15,
        c: float = 10_000.0,
        pool_size: int = 2000,
        ns_list: tuple[int, ...] = (50, 200, 500, 2000),
        slope_atol: float = 0.15,
    ) -> None:
        """
        Проверяет сходимость std_error(mean_FV) ∝ 1/√N (ЦПТ).

        Алгоритм:
          1. Генерируем pool_size независимых FV (каждый — случайная траектория доходности).
          2. Для каждого N из ns_list: std_error_N = std(FV[:N]) / √N.
          3. Регрессия log(std_error) ~ β·log(N) в МНК.
          4. По ЦПТ ожидаемый наклон β ≈ −0.5.

        Тест: |β − (−0.5)| < slope_atol.
        """
        name = "Монте-Карло: сходимость std_error ∝ 1/√N (ЦПТ)"

        fv_pool = np.empty(pool_size)
        for i in range(pool_size):
            rates = list(np.random.normal(r_mean, r_sigma, size=n_years))
            params = ProgramInput(
                n=n_years, age=30, sex="M",
                rates=rates,
                payment_mode="const",
                fix_payment=c,
                initial_salary=50_000.0,
                tax_deduction_rate=0.0,
            )
            prog = IIS3Program(params=params, life_table=None)
            prog.run()
            fv_pool[i] = prog.final_accumulation

        std_errors = [
            float(np.std(fv_pool[:N], ddof=1) / np.sqrt(N))
            for N in ns_list
        ]

        # OLS в лог-лог пространстве: log(se) = β·log(N) + const
        log_n = np.log(np.array(ns_list, dtype=float))
        log_se = np.log(np.array(std_errors))
        slope, intercept, r_sq, *_ = stats.linregress(log_n, log_se)

        passed = abs(slope - (-0.5)) < slope_atol

        self._add(TestResult(
            name=name, passed=passed,
            message=(f"наклон β={slope:.4f} (ожид. −0.5 ± {slope_atol}), R²={r_sq:.4f}"),
            details=dict(
                ns=list(ns_list),
                std_errors=std_errors,
                fitted_slope=float(slope),
                r_squared=float(r_sq),
            ),
        ))


# ══════════════════════════════════════════════════════════════════════════════
# Convenience entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_calibration(
    n_sims: int = 3000,
    alpha: float = 0.05,
    seed: int = 42,
) -> CalibrationSuite:
    """
    Запустить полный калибровочный прогон и напечатать отчёт.

    Parameters
    ----------
    n_sims : число реплик МК для стохастических тестов
    alpha  : уровень значимости для статистических критериев
    seed   : seed для воспроизводимости

    Returns
    -------
    CalibrationSuite с заполненными результатами
    """
    suite = CalibrationSuite(n_sims=n_sims, alpha=alpha, seed=seed)
    suite.run_all()
    print(suite.summary())
    return suite


if __name__ == "__main__":
    run_calibration()
