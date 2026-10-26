# Threat model — PhishShift ML-07

## Intended use
Helpdesk/security-awareness training to triage controlled simulated email/SMS text.

## Assets
- Synthetic corpus and model artifacts
- Scanner availability and displayed prediction
- User understanding of uncertainty and limitations

## Threats and mitigations
- Malicious URL execution: no network fetch, link preview, or browser automation.
- Secret collection: UI warns against real credentials and limits text to 10,000 characters.
- Template leakage: normalized/near-duplicate removal before campaign-group split.
- Overconfident prediction: abstain zone and explicit decision-support warning.
- False positives on legitimate urgent content: separate held-out metrics and acceptance test.
- Model evasion via wording/punctuation: controlled perturbation test log.
- Data privacy risk: synthetic-only corpus, no mailbox imports or storage of submitted scans.
## Out of scope
Live email gateway integration, URL reputation lookups, attachment scanning, sender authentication, and a guarantee that any message is safe.
