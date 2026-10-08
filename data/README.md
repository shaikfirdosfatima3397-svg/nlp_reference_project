# Dataset provenance

All tickets, names, emails and store policies in this folder were written specifically for this teaching project. They are synthetic, not collected customer data. example.com addresses are illustrative.

train.csv: 120 English tickets, 24 per category. test.csv: 40 separately written English tickets, 8 per category. The labels and split are checked in; the application never fits the classifier on test.csv. Synthetic holdout scores do not establish real-world performance. Some vocabulary overlap is intentional because the domain is shared.

extraction_test.json: eight entity fixtures. retrieval_test.json: twelve single-relevant-policy queries. demo_tickets.json: six UI examples, separate from train/test. policies.json: seven fictional policies, including keyword aliases for sparse retrieval. Aliases are curated features, not learned semantic understanding. Policy dates and conditions are fictional.

Before real use, replace the data with consented, anonymized tickets and approved policies. Preserve a separate final test set. Label edge cases, multi-intent messages, unsupported topics and non-English text.
