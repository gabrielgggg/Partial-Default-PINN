"""tables.py -- LaTeX tables of the report from RUN_DIR/eval.json and train.json.
Usage: python -m pinn.tables RUN_DIR OUT_DIR"""
import json
import os
import sys

ROWS = [("pd_freq", "frequency of partial default"), ("pd_mean_cond", "mean partial default, in default"),
        ("pd_sd_cond", "s.d. partial default, in default"), ("debt_y_mean", "mean debt / output"),
        ("debt_y_sd", "s.d. debt / output"), ("ds_y_mean", "mean debt service / output"),
        ("ds_y_sd", "s.d. debt service / output"), ("dd_y_mean", "mean debt due / output"),
        ("spread_sd", "s.d. spread"), ("y_persist", "persistence of output"), ("y_sd", "s.d. output"),
        ("dc_y_mean_cond", "mean defaulted coupons / output$^*$"), ("dc_y_sd_cond", "s.d. defaulted coupons / output$^*$"),
        ("spread_mean", "mean spread$^*$"), ("corr_spread_y", "corr(spread, output)$^*$"),
        ("corr_spread_debt", "corr(spread, debt)$^*$"), ("c_sd_rel", "s.d. consumption / s.d. output$^*$")]
OBJ = [("V", "value $V$"), ("q", "price $q$"), ("c", "consumption $c$"), ("d", "default intensity $d$"), ("S", "drift $S$")]


def main():
    run, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    E = json.load(open(os.path.join(run, "eval.json")))
    P, D = E["grids"]["paper"], E["grids"]["dense"]
    f = lambda v: f"{v:.3f}"
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{2}{c}{paper grid, $201\times 75$} & \multicolumn{2}{c}{dense grid, $2000\times 121$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"moment & FD & PINN & FD & PINN \\", r"\midrule"]
    for k, lab in ROWS:
        L.append(f"{lab} & {f(P['moments_fd'][k])} & {f(P['moments_pinn'][k])} & {f(D['moments_fd'][k])} & {f(D['moments_pinn'][k])} \\\\")
    L.append(r"\midrule")
    L.append(f"loss & {P['loss_fd']:.4f} & {P['loss_pinn']:.4f} & {D['loss_fd']:.4f} & {D['loss_pinn']:.4f} \\\\")
    L.append(f"mass at $B=0$ & {P['density']['atom0_fd']:.3f} & {P['density']['atom0_pinn']:.3f} & "
             f"{D['density']['atom0_fd']:.3f} & {D['density']['atom0_pinn']:.3f} \\\\")
    L.append(f"mean debt $E[B]$ & {P['density']['EB_fd']:.3f} & {P['density']['EB_pinn']:.3f} & "
             f"{D['density']['EB_fd']:.3f} & {D['density']['EB_pinn']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "moments.tex"), "w").write("\n".join(L) + "\n")
    e = lambda v: f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e")
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{paper grid, $201\times 75$} & \multicolumn{3}{c}{dense grid, $2000\times 121$} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"object & max & 99th pct. & ergodic mean & max & 99th pct. & ergodic mean \\", r"\midrule"]
    for k, lab in OBJ:
        a, b = P["errors"][k], D["errors"][k]
        L.append(f"{lab} & {e(a['max'])} & {e(a['p99_mass'])} & {e(a['erg'])} & {e(b['max'])} & {e(b['p99_mass'])} & {e(b['erg'])} \\\\")
    L.append(r"\midrule")
    L.append(f"ergodic density, $\\sum|g-g^{{FD}}|$ & \\multicolumn{{3}}{{c}}{{{P['density']['l1']:.3f}}} & \\multicolumn{{3}}{{c}}{{{D['density']['l1']:.3f}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "errors.tex"), "w").write("\n".join(L) + "\n")
    T = json.load(open(os.path.join(run, "train.json")))
    a = T["args"]
    vals = dict(train_min=T["train_seconds"] / 60, steps=a["outer"] * a["inner"], outer=a["outer"], inner=a["inner"],
                eval_paper=P["eval_seconds"], eval_dense=D["eval_seconds"],
                resV=P["residuals"]["erg_V"], resq=P["residuals"]["erg_q"], resVmax=P["residuals"]["max_V"], resqmax=P["residuals"]["max_q"])
    open(os.path.join(out, "numbers.tex"), "w").write("\n".join(
        [f"\\newcommand{{\\trainmin}}{{{vals['train_min']:.0f}}}", f"\\newcommand{{\\nsteps}}{{{vals['steps']:,}}}",
         f"\\newcommand{{\\nouter}}{{{vals['outer']}}}", f"\\newcommand{{\\ninner}}{{{vals['inner']}}}",
         f"\\newcommand{{\\evalpaper}}{{{vals['eval_paper']:.2f}}}", f"\\newcommand{{\\evaldense}}{{{vals['eval_dense']:.2f}}}",
         f"\\newcommand{{\\resV}}{{{e(vals['resV'])}}}", f"\\newcommand{{\\resq}}{{{e(vals['resq'])}}}",
         f"\\newcommand{{\\resVmax}}{{{e(vals['resVmax'])}}}", f"\\newcommand{{\\resqmax}}{{{e(vals['resqmax'])}}}"]) + "\n")
    print("tables written")


if __name__ == "__main__":
    main()
