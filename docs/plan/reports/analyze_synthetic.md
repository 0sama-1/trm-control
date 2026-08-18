# trm-control analysis report

Generated: 2026-08-18 06:07:16

## Metadata

- **csv**: /workspace/trm/trm_control_integration/results/synthetic.csv
- **n_runs**: 30
- **n_conditions**: 3
- **n_boot**: 2000
- **conf**: 0.95

## Per-condition summary

| condition | n | mean_return | std_return | sem | ci95_lo | ci95_hi | mean_wall_clock | mean_steps_per_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arch=ppo,model=dnn,env=pendulum,delay=none | 10 | -196.4910 | 29.5028 | 9.3296 | -214.0256 | -178.8192 | 30.0610 | 381.2810 |
| arch=ppo,model=trm,env=pendulum,delay=none | 10 | -182.0570 | 23.9139 | 7.5622 | -196.6757 | -169.0629 | 31.3750 | 401.8860 |
| arch=ppo,model=trm,env=pendulum,delay=none,H_cycles=3 | 10 | -157.0870 | 22.3380 | 7.0639 | -170.0908 | -144.1894 | 30.9630 | 430.6740 |

## Pairwise contrasts

Cohen's d (positive = condition_a > condition_b); paired t-test where seeds match, Welch otherwise.

| condition_a | condition_b | cohens_d | t_stat | p_value | test |
| --- | --- | --- | --- | --- | --- |
| arch=ppo,model=dnn,env=pendulum,delay=none | arch=ppo,model=trm,env=pendulum,delay=none | -0.5375 | -0.9390 | 0.3722 | paired |
| arch=ppo,model=dnn,env=pendulum,delay=none | arch=ppo,model=trm,env=pendulum,delay=none,H_cycles=3 | -1.5059 | -2.8153 | 0.0202 | paired |
| arch=ppo,model=trm,env=pendulum,delay=none | arch=ppo,model=trm,env=pendulum,delay=none,H_cycles=3 | -1.0791 | -2.8366 | 0.0195 | paired |

## Per-run wall-clock / throughput (plan §5.5)

| condition | n | mean_wall_clock_s | mean_steps_per_sec |
| --- | --- | --- | --- |
| arch=ppo,model=dnn,env=pendulum,delay=none | 10 | 30.0610 | 381.2810 |
| arch=ppo,model=trm,env=pendulum,delay=none | 10 | 31.3750 | 401.8860 |
| arch=ppo,model=trm,env=pendulum,delay=none,H_cycles=3 | 10 | 30.9630 | 430.6740 |

## Headline

Best mean return: **-157.0870** (arch=ppo,model=trm,env=pendulum,delay=none,H_cycles=3, n=10, 95% CI [-170.0908, -144.1894]).

