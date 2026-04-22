# Results

## Question

Can a conditional residual model widen bands in stressed regimes while staying calibrated at the 5% lower tail?

## Single Split

- Point forecast beats better naive: `False`
- Unconditional lower-tail breach rate: `0.0467`
- Unconditional Kupiec p-value: `0.6800`
- Stressed-only lower-tail breach rate: `0.0263`
- Stressed-only Kupiec p-value: `0.4630`

## Walk Forward

- Pooled unconditional breach rate: `0.0573`
- Pooled unconditional Kupiec p-value: `0.1253`
- Pooled stressed breach rate: `0.0538`
- Pooled stressed Kupiec p-value: `0.7282`

## Acceptance

- Pooled unconditional Kupiec passes: `True`
- Pooled stressed Kupiec passes: `True`
- Acceptance bar passed: `True`
