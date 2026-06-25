# AL benchmark — stats summary (1 seed(s): [1000], 19 usable datasets; rose1+degenerate excluded)

Paired analysis: per-policy vs Random(P0), datasets as the unit, seeds averaged first. **TIE** = |mean_diff| < cross-dataset SE; significance from Wilcoxon signed-rank + sign test.


## By final DSC vs by AUBC (budget-curve area)

| rank | by final-DSC | by AUBC |
|---|---|---|
| 1 | P5 Entropy+CoreSet (0.736) | P8 SAM-TypiClust (0.681) |
| 2 | P0 Random (0.730) | P5 Entropy+CoreSet (0.679) |
| 3 | P4 BADGE (0.726) | P0 Random (0.675) |
| 4 | P8 SAM-TypiClust (0.724) | P4 BADGE (0.674) |
| 5 | P7 SAM-CoreSet (0.723) | P7 SAM-CoreSet (0.671) |
| 6 | P2 BALD (0.722) | P2 BALD (0.657) |
| 7 | P1 Entropy (0.707) | P1 Entropy (0.654) |
| 8 | P3 CoreSet (0.705) | P3 CoreSet (0.650) |
| 9 | P9 PAAL (0.692) | P9 PAAL (0.642) |
| 10 | P6 SelUncertainty (0.650) | P6 SelUncertainty (0.627) |

## Per-policy summary (vs Random P0)

| policy | final | AUBC | @5% | Δfinal-vs-P0 | 95% CI | win/N | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|
| P5 Entropy+CoreSet | 0.736 | 0.679 | 0.637 | +0.0060 | [-0.0112,+0.0240] | 10/19 | 0.651 | **TIE** |
| P0 Random | 0.730 | 0.675 | 0.646 | — | — | — | — | baseline |
| P4 BADGE | 0.726 | 0.674 | 0.638 | -0.0040 | [-0.0156,+0.0081] | 9/19 | 0.490 | **TIE** |
| P8 SAM-TypiClust | 0.724 | 0.681 | 0.658 | -0.0066 | [-0.0247,+0.0127] | 6/19 | 0.243 | **TIE** |
| P7 SAM-CoreSet | 0.723 | 0.671 | 0.636 | -0.0078 | [-0.0178,+0.0014] | 7/19 | 0.156 | **TIE** |
| P2 BALD | 0.722 | 0.657 | 0.602 | -0.0089 | [-0.0308,+0.0119] | 9/19 | 0.768 | **TIE** |
| P1 Entropy | 0.707 | 0.654 | 0.622 | -0.0234 | [-0.0588,+0.0053] | 8/19 | 0.275 | **TIE** |
| P3 CoreSet | 0.705 | 0.650 | 0.600 | -0.0255 | [-0.0389,-0.0127] | 5/19 | 0.003 | sig<P0 |
| P9 PAAL | 0.692 | 0.642 | 0.596 | -0.0386 | [-0.0614,-0.0162] | 5/19 | 0.005 | sig<P0 |
| P6 SelUncertainty | 0.650 | 0.627 | 0.610 | -0.0802 | [-0.1711,-0.0201] | 5/19 | 0.003 | sig<P0 |

cross-policy SE (tie threshold) ≈ 0.0241. ns = not significant; sig requires Wilcoxon p<0.05 AND CI excluding 0.
