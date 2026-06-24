# Risk Label Construction

Labels are assigned via rule-based thresholds on six driving parameters. The logic is deterministic: every sample falls into exactly one of three categories based solely on its feature values.

## Low Risk (label = 0)

All of the following must hold:

| Parameter            | Condition           |
|----------------------|---------------------|
| speed                | ≤ 80 km/h           |
| brake_force          | ≤ 0.2               |
| acceleration         | \|a\| ≤ 2 m/s²      |
| steering_rate        | ≤ 0.5 deg/s         |
| following_distance   | ≥ 30 m              |
| blink_frequency      | 12 – 25 blinks/min  |

## High Risk (label = 2)

Any one of the following triggers a high-risk assignment:

| Condition                                         |
|---------------------------------------------------|
| speed ≥ 120 km/h **and** brake_force ≥ 0.7        |
| following_distance ≤ 10 m **and** speed ≥ 100 km/h|
| steering_rate ≥ 2.0 deg/s **and** speed ≥ 80 km/h |
| blink_frequency ≥ 35 blinks/min                   |

## Medium Risk (label = 1)

Samples that fail the low-risk criteria but do not trigger any high-risk condition. This is the residual class — no explicit thresholds define it.
