# Task 1 analysis summary

## 1. Benchmarks

| condition | gsm8k % | gsm8k change pp | hellaswag % | hellaswag raw acc % | hellaswag change pp | hellaswag raw change pp | boolq % | boolq change pp | arc_easy % | arc_easy raw acc % | arc_easy change pp | arc_easy raw change pp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 43.37 | +0.00 | 47.32 | 37.56 | +0.00 | +0.00 | 64.37 | +0.00 | 56.06 | 60.90 | +0.00 | +0.00 |
| full_ft_u0625 | 57.70 | +14.33 | 45.72 | 36.93 | -1.60 | -0.63 | 61.65 | -2.72 | 61.49 | 65.36 | +5.43 | +4.46 |
| full_ft_u1250 | 64.37 | +21.00 | 45.86 | 37.40 | -1.46 | -0.16 | 60.61 | -3.76 | 62.75 | 66.25 | +6.69 | +5.35 |
| l2sp_u1250 | 63.99 | +20.62 | 46.56 | 37.45 | -0.76 | -0.11 | 65.29 | +0.92 | 62.96 | 66.16 | +6.90 | +5.26 |
| lora4_u1250 | 60.35 | +16.98 | 47.24 | 37.73 | -0.08 | +0.17 | 66.67 | +2.29 | 63.97 | 66.16 | +7.91 | +5.26 |
| lora16_u1250 | 62.55 | +19.18 | 47.44 | 37.81 | +0.12 | +0.25 | 70.98 | +6.61 | 63.80 | 66.16 | +7.74 | +5.26 |
| oplora16_u1250 | 62.55 | +19.18 | 47.14 | 37.79 | -0.18 | +0.23 | 70.89 | +6.51 | 63.68 | 65.66 | +7.62 | +4.76 |

Paired item bootstrap, 10000 resamples, seed 6304, primary metric:

| A | B | benchmark | A - B pp | 95% CI pp | only A correct | only B correct | n |
|---|---|---|---|---|---|---|---|
| full_ft_u0625 | baseline | gsm8k | +14.33 | [+11.30, +17.36] | 321 | 132 | 1319 |
| full_ft_u0625 | baseline | hellaswag | -1.60 | [-2.20, -1.03] | 369 | 530 | 10042 |
| full_ft_u0625 | baseline | boolq | -2.72 | [-4.77, -0.70] | 543 | 632 | 3270 |
| full_ft_u0625 | baseline | arc_easy | +5.43 | [+3.96, +6.94] | 239 | 110 | 2376 |
| full_ft_u1250 | baseline | gsm8k | +21.00 | [+17.89, +24.11] | 386 | 109 | 1319 |
| full_ft_u1250 | baseline | hellaswag | -1.46 | [-2.04, -0.89] | 371 | 518 | 10042 |
| full_ft_u1250 | baseline | boolq | -3.76 | [-5.84, -1.71] | 541 | 664 | 3270 |
| full_ft_u1250 | baseline | arc_easy | +6.69 | [+5.18, +8.25] | 264 | 105 | 2376 |
| l2sp_u1250 | baseline | gsm8k | +20.62 | [+17.59, +23.65] | 370 | 98 | 1319 |
| l2sp_u1250 | baseline | hellaswag | -0.76 | [-1.30, -0.21] | 356 | 432 | 10042 |
| l2sp_u1250 | baseline | boolq | +0.92 | [-1.04, +2.88] | 554 | 524 | 3270 |
| l2sp_u1250 | baseline | arc_easy | +6.90 | [+5.43, +8.42] | 258 | 94 | 2376 |
| lora4_u1250 | baseline | gsm8k | +16.98 | [+13.80, +20.24] | 359 | 135 | 1319 |
| lora4_u1250 | baseline | hellaswag | -0.08 | [-0.55, +0.39] | 281 | 289 | 10042 |
| lora4_u1250 | baseline | boolq | +2.29 | [+0.43, +4.16] | 523 | 448 | 3270 |
| lora4_u1250 | baseline | arc_easy | +7.91 | [+6.44, +9.47] | 273 | 85 | 2376 |
| lora16_u1250 | baseline | gsm8k | +19.18 | [+16.07, +22.44] | 368 | 115 | 1319 |
| lora16_u1250 | baseline | hellaswag | +0.12 | [-0.38, +0.62] | 327 | 315 | 10042 |
| lora16_u1250 | baseline | boolq | +6.61 | [+4.80, +8.44] | 578 | 362 | 3270 |
| lora16_u1250 | baseline | arc_easy | +7.74 | [+6.27, +9.26] | 262 | 78 | 2376 |
| oplora16_u1250 | baseline | gsm8k | +19.18 | [+16.07, +22.37] | 370 | 117 | 1319 |
| oplora16_u1250 | baseline | hellaswag | -0.18 | [-0.69, +0.32] | 317 | 335 | 10042 |
| oplora16_u1250 | baseline | boolq | +6.51 | [+4.68, +8.35] | 579 | 366 | 3270 |
| oplora16_u1250 | baseline | arc_easy | +7.62 | [+6.14, +9.18] | 268 | 87 | 2376 |
| full_ft_u1250 | full_ft_u0625 | gsm8k | +6.67 | [+4.32, +9.02] | 175 | 87 | 1319 |
| full_ft_u1250 | full_ft_u0625 | hellaswag | +0.14 | [-0.21, +0.48] | 158 | 144 | 10042 |
| full_ft_u1250 | full_ft_u0625 | boolq | -1.04 | [-1.71, -0.37] | 46 | 80 | 3270 |
| full_ft_u1250 | full_ft_u0625 | arc_easy | +1.26 | [+0.46, +2.06] | 60 | 30 | 2376 |
| lora4_u1250 | lora16_u1250 | gsm8k | -2.20 | [-4.32, -0.08] | 86 | 115 | 1319 |
| lora4_u1250 | lora16_u1250 | hellaswag | -0.20 | [-0.51, +0.11] | 115 | 135 | 10042 |
| lora4_u1250 | lora16_u1250 | boolq | -4.31 | [-5.47, -3.15] | 121 | 262 | 3270 |
| lora4_u1250 | lora16_u1250 | arc_easy | +0.17 | [-0.63, +0.97] | 48 | 44 | 2376 |
| full_ft_u1250 | l2sp_u1250 | gsm8k | +0.38 | [-1.44, +2.12] | 75 | 70 | 1319 |
| full_ft_u1250 | l2sp_u1250 | hellaswag | -0.71 | [-1.07, -0.34] | 135 | 206 | 10042 |
| full_ft_u1250 | l2sp_u1250 | boolq | -4.68 | [-5.72, -3.67] | 75 | 228 | 3270 |
| full_ft_u1250 | l2sp_u1250 | arc_easy | -0.21 | [-1.14, +0.72] | 58 | 63 | 2376 |
| lora16_u1250 | oplora16_u1250 | gsm8k | +0.00 | [-1.59, +1.59] | 58 | 58 | 1319 |
| lora16_u1250 | oplora16_u1250 | hellaswag | +0.30 | [+0.07, +0.53] | 88 | 58 | 10042 |
| lora16_u1250 | oplora16_u1250 | boolq | +0.09 | [-0.55, +0.73] | 60 | 57 | 3270 |
| lora16_u1250 | oplora16_u1250 | arc_easy | +0.13 | [-0.46, +0.72] | 27 | 24 | 2376 |
| full_ft_u1250 | lora16_u1250 | gsm8k | +1.82 | [-0.45, +4.09] | 125 | 101 | 1319 |
| full_ft_u1250 | lora16_u1250 | hellaswag | -1.58 | [-2.07, -1.10] | 229 | 388 | 10042 |
| full_ft_u1250 | lora16_u1250 | boolq | -10.37 | [-12.17, -8.56] | 298 | 637 | 3270 |
| full_ft_u1250 | lora16_u1250 | arc_easy | -1.05 | [-2.23, +0.13] | 91 | 116 | 2376 |

Per-example flags reproduce scores.json exactly for all conditions and benchmarks.

## 2. GSM8K extraction vs correctness

| condition | correct | wrong | wrong, extracted | of which no digit | no digit: gold is last number | unextractable | truncated | truncated wrong | strict rate | strict acc | flexible - strict pp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 572 | 747 | 747 | 334 | 199 | 0 | 11 | 11 | 0.0000 | 0.0000 | +43.37 |
| full_ft_u0625 | 761 | 558 | 558 | 0 | 0 | 0 | 18 | 18 | 0.0000 | 0.0000 | +57.70 |
| full_ft_u1250 | 849 | 470 | 470 | 0 | 0 | 0 | 10 | 9 | 0.0000 | 0.0000 | +64.37 |
| l2sp_u1250 | 844 | 475 | 475 | 0 | 0 | 0 | 7 | 7 | 0.0000 | 0.0000 | +63.99 |
| lora4_u1250 | 796 | 523 | 523 | 0 | 0 | 0 | 12 | 12 | 0.0000 | 0.0000 | +60.35 |
| lora16_u1250 | 825 | 494 | 494 | 0 | 0 | 0 | 5 | 5 | 0.0000 | 0.0000 | +62.55 |
| oplora16_u1250 | 825 | 494 | 494 | 0 | 0 | 0 | 6 | 6 | 0.0000 | 0.0000 | +62.55 |

Items gained and lost against the baseline:

| condition | gained | base trunc | cond trunc | neither | base no digit | base no digit, gold last | lost | base trunc | cond trunc | neither |
|---|---|---|---|---|---|---|---|---|---|---|
| full_ft_u0625 | 321 | 1 | 0 | 320 | 181 | 145 | 132 | 0 | 2 | 130 |
| full_ft_u1250 | 386 | 2 | 1 | 383 | 212 | 164 | 109 | 0 | 1 | 108 |
| l2sp_u1250 | 370 | 2 | 0 | 368 | 199 | 154 | 98 | 0 | 1 | 97 |
| lora4_u1250 | 359 | 3 | 0 | 356 | 192 | 149 | 135 | 0 | 2 | 133 |
| lora16_u1250 | 368 | 2 | 0 | 366 | 195 | 157 | 115 | 0 | 1 | 114 |
| oplora16_u1250 | 370 | 3 | 0 | 367 | 198 | 154 | 117 | 0 | 0 | 117 |

## 3. Prediction balance and normalization

| condition | gold yes % | pred yes % | acc % | acc gold yes % | acc gold no % |
|---|---|---|---|---|---|
| baseline | 62.17 | 57.74 | 64.37 | 67.78 | 58.77 |
| full_ft_u0625 | 62.17 | 31.71 | 61.65 | 44.66 | 89.57 |
| full_ft_u1250 | 62.17 | 29.57 | 60.61 | 42.11 | 91.03 |
| l2sp_u1250 | 62.17 | 38.78 | 65.29 | 53.27 | 85.04 |
| lora4_u1250 | 62.17 | 47.06 | 66.67 | 61.04 | 75.91 |
| lora16_u1250 | 62.17 | 58.17 | 70.98 | 73.44 | 66.94 |
| oplora16_u1250 | 62.17 | 56.12 | 70.89 | 71.72 | 69.52 |

| benchmark | condition | acc % | acc_norm % | acc change pp | acc_norm change pp | norm - raw pp |
|---|---|---|---|---|---|---|
| hellaswag | baseline | 37.56 | 47.32 | +0.00 | +0.00 | +0.00 |
| hellaswag | full_ft_u0625 | 36.93 | 45.72 | -0.63 | -1.60 | -0.98 |
| hellaswag | full_ft_u1250 | 37.40 | 45.86 | -0.16 | -1.46 | -1.30 |
| hellaswag | l2sp_u1250 | 37.45 | 46.56 | -0.11 | -0.76 | -0.65 |
| hellaswag | lora4_u1250 | 37.73 | 47.24 | +0.17 | -0.08 | -0.25 |
| hellaswag | lora16_u1250 | 37.81 | 47.44 | +0.25 | +0.12 | -0.13 |
| hellaswag | oplora16_u1250 | 37.79 | 47.14 | +0.23 | -0.18 | -0.41 |
| arc_easy | baseline | 60.90 | 56.06 | +0.00 | +0.00 | +0.00 |
| arc_easy | full_ft_u0625 | 65.36 | 61.49 | +4.46 | +5.43 | +0.97 |
| arc_easy | full_ft_u1250 | 66.25 | 62.75 | +5.35 | +6.69 | +1.35 |
| arc_easy | l2sp_u1250 | 66.16 | 62.96 | +5.26 | +6.90 | +1.64 |
| arc_easy | lora4_u1250 | 66.16 | 63.97 | +5.26 | +7.91 | +2.65 |
| arc_easy | lora16_u1250 | 66.16 | 63.80 | +5.26 | +7.74 | +2.48 |
| arc_easy | oplora16_u1250 | 65.66 | 63.68 | +4.76 | +7.62 | +2.86 |

## 4. Learning and regularization

| run | update | displacement | mean SFT last 50 | Omega | 0.5 disp^2 | lambda Omega | total loss | lambda Omega / mean SFT |
|---|---|---|---|---|---|---|---|---|
| full_ft | 625 | 10.0270 | - | - | - | - | - | - |
| full_ft | 1250 | 10.6650 | 0.18422 | - | - | - | - | - |
| l2sp | 1250 | 6.2686 | 0.18419 | 19.6478 | 19.6478 | 0.01965 | 0.18378 | 0.1067 |

| run | mean SFT last 50 |
|---|---|
| full_ft | 0.18422 |
| l2sp | 0.18419 |
| lora4 | 0.21174 |
| lora16 | 0.19897 |
| oplora16 | 0.20003 |

Figure: figures/task1/learning_curves.png

## 5. Resources

| run | trainable params | train h | wall h | s/update | peak alloc GiB | peak reserved GiB | kept files MB | recovery state MB | started UTC | shared VM with |
|---|---|---|---|---|---|---|---|---|---|---|
| full_ft | 596,049,920 | 2.34 | 2.42 | 6.75 | 11.70 | 12.74 | 4,768.5 | 7,153.1 | 10-29 12:10 | l2sp |
| l2sp | 596,049,920 | 2.40 | 2.47 | 6.91 | 12.80 | 14.42 | 2,384.2 | 7,153.2 | 10-29 12:12 | full_ft, lora4 |
| lora4 | 2,523,136 | 2.90 | 2.90 | 8.35 | 3.14 | 5.92 | 10.1 | 30.9 | 10-29 14:36 | l2sp, lora16 |
| lora16 | 10,092,544 | 2.78 | 2.78 | 8.00 | 3.26 | 6.04 | 40.4 | 121.7 | 10-29 14:40 | lora4 |
| oplora16 | 10,092,544 | 3.66 | 3.67 | 10.55 | 3.56 | 6.35 | 80.8 | 444.8 | 10-29 17:31 | none |

| overhead | value |
|---|---|
| L2-SP theta_0 bytes | 1,192,099,840 |
| L2-SP theta_0 build s | 0.0346 |
| L2-SP penalty s/update | 0.1759 |
| OPLoRA SVD s | 34.54 |
| OPLoRA basis bytes | 322,961,408 |
| OPLoRA export s | 0.1857 |

## 6. Diagnostics

Direction-agnostic rho_k: k/d_out = 128/1024 = 0.125.

| condition | layer | rho_k | eta | R math median | R math IQR | R non-math median | R non-math IQR | math/non-math |
|---|---|---|---|---|---|---|---|---|
| full_ft_u0625 | 0 | 0.1338 | 0.01327 | 0.05213 | 0.00034 | 0.02709 | 0.00132 | 1.925 |
| full_ft_u0625 | 13 | 0.1484 | 0.01541 | 0.05712 | 0.00097 | 0.02990 | 0.00436 | 1.910 |
| full_ft_u0625 | 27 | 0.1157 | 0.01247 | 0.01968 | 0.00289 | 0.02841 | 0.00211 | 0.693 |
| full_ft_u1250 | 0 | 0.1336 | 0.01372 | 0.05308 | 0.00036 | 0.02771 | 0.00140 | 1.916 |
| full_ft_u1250 | 13 | 0.1483 | 0.01605 | 0.05764 | 0.00096 | 0.03051 | 0.00415 | 1.889 |
| full_ft_u1250 | 27 | 0.1151 | 0.01296 | 0.01956 | 0.00296 | 0.02888 | 0.00234 | 0.677 |
| l2sp_u1250 | 0 | 0.1331 | 0.01119 | 0.04522 | 0.00034 | 0.02382 | 0.00100 | 1.898 |
| l2sp_u1250 | 13 | 0.1496 | 0.01370 | 0.04773 | 0.00102 | 0.02728 | 0.00418 | 1.749 |
| l2sp_u1250 | 27 | 0.107 | 0.00788 | 0.01848 | 0.00250 | 0.02574 | 0.00173 | 0.718 |
| lora4_u1250 | 0 | 0.134 | 0.00830 | 0.06938 | 0.00114 | 0.01579 | 0.00358 | 4.394 |
| lora4_u1250 | 13 | 0.1349 | 0.01108 | 0.09558 | 0.00207 | 0.01356 | 0.00818 | 7.046 |
| lora4_u1250 | 27 | 0.1029 | 0.00611 | 0.00405 | 0.00049 | 0.01219 | 0.00375 | 0.332 |
| lora16_u1250 | 0 | 0.1334 | 0.01424 | 0.05038 | 0.00117 | 0.02314 | 0.00268 | 2.178 |
| lora16_u1250 | 13 | 0.1416 | 0.01903 | 0.08941 | 0.00240 | 0.02438 | 0.00595 | 3.667 |
| lora16_u1250 | 27 | 0.115 | 0.01110 | 0.00938 | 0.00072 | 0.01948 | 0.00634 | 0.481 |
| oplora16_u1250 | 0 | 9.239e-10 | 0.01467 | 0.05499 | 0.00101 | 0.02538 | 0.00270 | 2.166 |
| oplora16_u1250 | 13 | 9.212e-10 | 0.01947 | 0.09969 | 0.00186 | 0.02731 | 0.00731 | 3.650 |
| oplora16_u1250 | 27 | 1.695e-09 | 0.00905 | 0.01151 | 0.00096 | 0.01580 | 0.00435 | 0.729 |

| OPLoRA constraint, 196 targets, k 128 | max | median |
|---|---|---|
| norm(U_k^T dW) / norm(dW), Frobenius | 0.000124 | 3.42e-05 |
| norm(dW V_k) / norm(dW), Frobenius | 0.00018 | 0.000159 |

| merge check | FP16 max | FP16 mean | FP16 argmax | FP32 max | FP32 mean | FP32 argmax |
|---|---|---|---|---|---|---|
| lora4_u1250 | 0.1533 | 0.006572 | 0.9976 | 0.00013 | 5.421e-06 | 1 |
| lora16_u1250 | 0.1416 | 0.006206 | 0.9952 | 0.0001144 | 5.148e-06 | 1 |
| oplora16_u1250 | 0.1611 | 0.006193 | 1 | 0.0001335 | 5.223e-06 | 1 |

Figure: figures/task1/diagnostics.png

## 7. Qualitative code checks

| condition | responses | mean tokens | truncated | JSON objects | JSON prompts | fenced | JSON truncated |
|---|---|---|---|---|---|---|---|
| baseline | 24 | 102.0 | 1 | 8 | 8 | 8 | 0 |
| full_ft_u0625 | 24 | 244.8 | 8 | 2 | 8 | 0 | 2 |
| full_ft_u1250 | 24 | 241.8 | 7 | 3 | 8 | 0 | 1 |
| l2sp_u1250 | 24 | 222.0 | 7 | 2 | 8 | 0 | 3 |
| lora4_u1250 | 24 | 110.2 | 2 | 3 | 8 | 0 | 1 |
| lora16_u1250 | 24 | 208.8 | 6 | 2 | 8 | 0 | 2 |
| oplora16_u1250 | 24 | 188.5 | 5 | 2 | 8 | 0 | 3 |

